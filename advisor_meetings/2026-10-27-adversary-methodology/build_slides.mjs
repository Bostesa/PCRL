import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const dir=path.dirname(new URL(import.meta.url).pathname);
const build=path.join(dir,'.build');
const out=path.join(dir,'output');
await fs.mkdir(build,{recursive:true}); await fs.mkdir(out,{recursive:true});
const skill=process.env.SKILL_DIR;
const runtimePython=process.env.RUNTIME_PYTHON;
if(!skill || !runtimePython)throw new Error('Set SKILL_DIR and RUNTIME_PYTHON. See README.');
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Arial'});
const content=JSON.parse(await fs.readFile(path.join(dir,'deck_content.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
function text(s,v,x,y,w,h,size=25,bold=false,color='#172A3A') {
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=v;
 sh.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
 return sh;
}
for(let i=0;i<content.slides.length;i++){
 const c=content.slides[i],s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,c.title,64,38,1152,64,39,true);
 text(s,c.lead,64,113,1152,70,i===1?25:24);
 if(c.table){
  const vals=[c.table.headers,...c.table.rows];
  const rowHeights=i===0?[54,64,86,98,102]:i===2?[62,100,100,100,100]:i===3?[70,78,92,92,78]:[54,83,75,91,83,93];
  const top=i===4?177:193;
  const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:64,top,width:1152,height:rowHeights.reduce((a,b)=>a+b,0),columnWidths:c.table.widths,values:vals});
  t.borders.assign({style:'solid',fill:'#C4CDD5',width:0.6});
  for(let r=0;r<vals.length;r++){
   t.rows[r].height=rowHeights[r];
   for(let j=0;j<vals[r].length;j++){
    const cell=t.getCell(r,j);
    cell.fill=r===0?'#1B354B':'#FFFFFF';
    cell.text.style={typeface:font,fontSize:r===0?22:(i===0?23:21),bold:r===0,color:r===0?'#FFFFFF':'#172A3A',autoFit:'none'};
   }
  }
  if(c.bottom)text(s,c.bottom,64,i===2?662:626,1152,i===2?46:72,i===2?20:23);
 }else{
  const ys=[204,310,425,550];
  c.blocks.forEach((b,k)=>{
   text(s,b.heading,64,ys[k],280,66,25,true);
   text(s,b.text,362,ys[k],854,105,24);
  });
 }
 s.speakerNotes.textFrame.setText(c.notes);
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
execFileSync(runtimePython,[path.join(dir,'document_properties.py'),candidate]);
const finalPath=path.join(out,'adversary_methodology.pptx');
await finalizePresentation({workspaceDir:dir,candidatePath:candidate,finalPath,pythonExecutable:runtimePython,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[1,3,4,5].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:5,requiredNativeTableOwnerSlides:[1,3,4,5],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'presentation.validation.json')});
console.log('Wrote',finalPath,'font',font);
