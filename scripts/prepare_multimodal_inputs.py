"""Use the official processor and embedding lookup for the fixed BPU shapes."""
import numpy as np
import torch

def prepare(model, value):
    features=model[0].preprocess([value],processing_kwargs={'image':{'max_soft_tokens':70}})
    auto=model[0].auto_model
    ids=features['input_ids']
    image_mask,video_mask,audio_mask=auto.get_placeholder_mask(ids)
    assert not video_mask.any() and not audio_mask.any(), 'This adapter covers text and one image'
    length=ids.shape[1]
    assert length<=128, f'Input exceeds fixed 128 tokens: {length}'
    clean=torch.where(image_mask,auto.config.text_config.pad_token_id,ids)
    with torch.no_grad():
        embeds=auto.get_input_embeddings()(clean).numpy()
    valid=features['attention_mask'].float().numpy()
    pixels=None
    if 'pixel_values' in features:
        assert int(image_mask.sum())==64
        assert features['pixel_values'].shape==(1,630,768),features['pixel_values'].shape
        pixels=features['pixel_values'].transpose(1,2).unsqueeze(2).numpy()
    return {'embeds':embeds,'valid':valid,'image_mask':image_mask.numpy(),
            'pixels':pixels,'positions':features.get('image_position_ids'),'tokens':length}

def fuse(prepared, vision_features=None):
    embeds=prepared['embeds'].copy()
    if vision_features is not None:
        assert vision_features.shape==(64,512)
        embeds[prepared['image_mask']]=vision_features
    else:
        assert not prepared['image_mask'].any(), 'Missing image features'
    padding=128-embeds.shape[1]
    x=np.pad(embeds,((0,0),(0,padding),(0,0))).transpose(0,2,1)[:,:,None,:]
    valid=np.pad(prepared['valid'],((0,0),(0,padding)))[:,None,None,:]
    return [np.ascontiguousarray(x,np.float32),np.ascontiguousarray(valid,np.float32)]

if __name__=='__main__':
    import json,os
    from pathlib import Path
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    model=SentenceTransformer(str(root/'model'),device='cpu',local_files_only=True,
        config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'dtype':torch.float32}).eval()
    refs=np.load(root/'bpu/results/vision_reference_features.npy')
    results=[]
    for i,(name,caption) in enumerate([('GoldenGate.png','The Golden Gate Bridge in San Francisco.'),
                                     ('kitchen_painting.jpg','A painting of a kitchen interior.')]):
        for mixed in [False,True]:
            value={'image':Image.open(root/'samples'/name).convert('RGB')}
            if mixed:value['text']=caption+' <|image|>'
            p=prepare(model,value)
            x,valid=fuse(p,refs[i])
            label=f'{i:03d}_'+('mixed' if mixed else 'image')
            expected=np.load(root/f'bpu/results/multimodal_semantic_diagnosis/{label}_fusion_inputs.npz')
            n=expected['inputs_embeds'].shape[1]
            error=float(np.max(np.abs(x.squeeze(2).transpose(0,2,1)[:,:n]-expected['inputs_embeds'])))
            assert error==0 and np.array_equal(valid.reshape(1,128)[:,:n],expected['attention_mask'])
            pixels=np.fromfile(root/f'bpu/calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(p['pixels'].shape)
            assert np.array_equal(pixels,p['pixels'])
            results.append({'case':label,'tokens':p['tokens'],'fusion_input_max_abs_error':error,'pixel_bytes_equal':True})
    out=root/'bpu/results/multimodal_input_preparation.json'
    out.write_text(json.dumps({'status':'complete','checks':results},indent=2))
    print(out,flush=True)
