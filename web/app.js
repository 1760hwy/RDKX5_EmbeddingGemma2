const $ = id => document.getElementById(id);
let sample = null, imageData = null, busy = false, ready = false;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function node(tag, cls, text) { const el=document.createElement(tag); if(cls)el.className=cls; if(text!==undefined)el.textContent=text; return el; }
function scrollEnd(){ $('messages').scrollTop=$('messages').scrollHeight; }
function refreshSend(){ $('send').disabled=busy||!ready; $('send').textContent=busy?'正在测试…':'开始测试 ↑'; }
function clearImage(){sample=null;imageData=null;$('file').value='';$('preview').replaceChildren();$('preview').hidden=true;}
function preview(src, label){$('preview').replaceChildren();const img=node('img');img.src=src;img.alt='待测试图片';$('preview').append(img,node('span','',label));const remove=node('button','','移除');remove.type='button';remove.onclick=clearImage;$('preview').append(remove);$('preview').hidden=false;}
function selectSample(name){sample=name;imageData=null;preview('/samples/'+name,{bridge:'金门大桥',kitchen:'厨房绘画',cat:'猫咪帧'}[name]);$('prompt').focus();}
document.querySelectorAll('[data-sample]').forEach(b=>b.onclick=()=>selectSample(b.dataset.sample));
$('text-example').onclick=()=>{$('prompt').value='一只小猫正在打盹';clearImage();$('prompt').focus();};
$('image-example').onclick=()=>{selectSample('bridge');$('prompt').value='';};
$('mixed-example').onclick=()=>{selectSample('kitchen');$('prompt').value='A painting of a kitchen interior.';};
$('file').onchange=()=>{const file=$('file').files[0];if(!file)return;if(file.size>5*1024*1024){alert('图片最大5MB，请缩小后再上传。');return;}const reader=new FileReader();reader.onload=()=>{imageData=reader.result;sample=null;preview(imageData,file.name);};reader.readAsDataURL(file);};
$('new-test').onclick=()=>{if(busy)return;$('messages').querySelectorAll('.message').forEach(m=>m.remove());$('welcome').hidden=false;clearImage();$('prompt').value='';};
function showResult(container,r){
  const card=node('section','result');card.append(node('div','result-head','模型结果 · '+{text:'文本',image:'图片',mixed:'图文混合'}[r.kind]));
  card.append(node('h3','', '最接近的候选：'+r.ranking[0].text));
  const rows=node('div','scores');r.ranking.forEach(item=>{const row=node('div','score');row.append(node('span','',item.text),node('span','number',item.cosine.toFixed(4)));rows.append(row);});card.append(rows);
  const metrics=node('div','metrics');[['输入推理',(r.input_timing.wall_ms/1000).toFixed(2)+' 秒'],['本次总耗时',(r.total_ms/1000).toFixed(2)+' 秒'],['向量维度',r.dimension+' 维'],['有效 tokens',String(r.input_timing.tokens)]].forEach(([label,val])=>{const m=node('div','metric',label);m.append(node('b','',val));metrics.append(m);});card.append(metrics);
  card.append(node('small','',`相似度是向量余弦，不是准确率。候选缓存命中 ${r.candidate_cache_hits}/${r.candidate_count}；首次候选编码计入总耗时。`));
  const details=node('details');details.append(node('summary','','查看向量与推理明细'),node('pre','',JSON.stringify({norm:r.norm,vector_preview:r.embedding.slice(0,16),stages:r.input_timing.stages},null,2)));card.append(details);
  const download=node('button','download','下载完整768维向量 JSON');download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download='embedding-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};card.append(download);container.replaceChildren(card);scrollEnd();
}
async function status(){try{const response=await fetch('/api/status');const s=await response.json();ready=s.status==='ready';$('status').classList.toggle('ready',ready);$('status-text').textContent=s.status==='failed'?s.message:ready?(s.active_job?'正在推理':'板端模型就绪'):s.message;refreshSend();}catch{ready=false;$('status-text').textContent='无法连接板端';refreshSend();}}
$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!busy&&ready)$('form').requestSubmit();}};
$('form').onsubmit=async e=>{
  e.preventDefault();if(busy||!ready)return;const text=$('prompt').value.trim();const candidates=$('candidates').value.split('\n').map(x=>x.trim()).filter(Boolean);if(!text&&!sample&&!imageData){$('prompt').focus();return;}if(candidates.length<1||candidates.length>12){alert('请输入1至12条候选描述。');return;}
  const payload={text,candidates,sample,image:imageData};const user=node('div','message user');const bubble=node('div','bubble');if(sample||imageData){const img=node('img');img.alt='本次测试图片';img.src=imageData||'/samples/'+sample;bubble.append(img);}bubble.append(node('span','',text||'测试这张图片'));user.append(bubble);const answer=node('div','message');const pending=node('div','pending','提交测试…');answer.append(pending);$('welcome').hidden=true;$('messages').append(user,answer);busy=true;refreshSend();scrollEnd();
  try{const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const job=await response.json();if(!response.ok)throw new Error(job.error||'提交失败');let done=false;for(let i=0;i<180;i++){await sleep(1000);const resp=await fetch('/api/jobs/'+job.id);const current=await resp.json();if(!resp.ok)throw new Error(current.error||'读取测试失败');if(current.status==='failed')throw new Error(current.error);if(current.status==='complete'){showResult(answer,current.result);done=true;break;}pending.textContent=current.progress+' · 请等待';}if(!done)throw new Error('等待超过3分钟，板端任务可能仍在运行。请查看状态，勿连续重复提交。');$('prompt').value='';clearImage();}
  catch(error){pending.classList.add('error');pending.textContent=error.message;scrollEnd();}
  finally{busy=false;refreshSend();}
};
status();setInterval(status,3000);
