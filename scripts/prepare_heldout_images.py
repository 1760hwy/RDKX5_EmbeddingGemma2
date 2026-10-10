"""Original-model references from existing, uncalibrated cat video frames.

Four frames from one clip are correlated; they are not four independent scenes.
This is an image test only, not acceptance of the model's video input support.
"""
import os,json,hashlib,time,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image,ImageOps
from sentence_transformers import SentenceTransformer
root=Path(__file__).resolve().parents[1]
square='--square-letterbox' in sys.argv
out=root/'bpu/vision_trim576'/('heldout_images_letterbox' if square else 'heldout_images')
out.mkdir(exist_ok=True)
report={'status':'running','started_at':time.time(),'cases':[],
        'preprocessing':'512x512 RGB bicubic letterbox, black background' if square else 'original image',
        'scope':'uncalibrated images, four correlated frames from one cat clip; not video acceptance'}
def save():(out/'references.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
try:
    save();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    model=SentenceTransformer(str(root/'model'),device='cpu',local_files_only=True,
        config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'eager'}).eval()
    auto=model[0].auto_model
    options={'image':{'max_soft_tokens':70}}
    base=model[0].preprocess([{'image':Image.open(root/'samples/GoldenGate.png').convert('RGB')}],processing_kwargs=options)
    positions=base['image_position_ids']
    keep=~(positions==-1).all(-1)[0]
    assert int(keep.sum())==576
    for i in range(1,5):
        source=root/f'samples/prepared/video_frames/frame_{i:02d}.png'
        image=Image.open(source).convert('RGB')
        if square:
            image=ImageOps.pad(image,(512,512),method=Image.Resampling.BICUBIC,color=(0,0,0))
            image.save(out/f'cat_frame_{i:02d}_letterbox.png')
        for kind in ['image','mixed']:
            value={'image':image}
            if kind=='mixed':value['text']='A cat dozing on a sofa. <|image|>'
            prepared=model[0].preprocess([value],processing_kwargs=options)
            assert torch.equal(prepared['image_position_ids'],positions), 'Heldout image needs another positional graph'
            mask,_,_=auto.get_placeholder_mask(prepared['input_ids'])
            assert int(mask.sum())==64
            captured={}
            def hook(module,args,kwargs):
                captured['inputs_embeds']=kwargs['inputs_embeds'].detach().numpy().copy()
                captured['attention_mask']=kwargs['attention_mask'].detach().numpy().copy()
            handle=auto.language_model.register_forward_pre_hook(hook,with_kwargs=True)
            try:
                reference=model.encode([value],normalize_embeddings=True,show_progress_bar=False,processing_kwargs=options)[0]
            finally:
                handle.remove()
            label=f'cat_frame_{i:02d}_{kind}'
            assert captured['inputs_embeds'].shape[1]<=128
            np.savez(out/f'{label}.npz',**captured,image_mask=mask.numpy(),reference=reference,
                pixels=prepared['pixel_values'][:,keep].transpose(1,2).unsqueeze(2).numpy())
            row={'case':label,'source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'processed_image':str((out/f'cat_frame_{i:02d}_letterbox.png').relative_to(root)) if square else str(source.relative_to(root)),
                 'tokens':captured['inputs_embeds'].shape[1],'expected_caption_indices':[2,6]}
            report['cases'].append(row);save();print(row,flush=True)
    report.update(status='complete',finished_at=time.time());save()
except Exception as error:
    report.update(status='failed',error=repr(error));save();raise
