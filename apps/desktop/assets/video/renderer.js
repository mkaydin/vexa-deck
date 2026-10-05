/* Local VEXA renderer. Pixel shader/palettes and ASCII ramp attribution:
   third_party/video-art/{README.md,LICENSE.txt}. No remote dependencies. */
'use strict';
(() => {
  const video = document.getElementById('source');
  const pixel = document.getElementById('pixel');
  const ascii = document.getElementById('ascii');
  const context = ascii.getContext('2d', {alpha: false});
  const sample = document.createElement('canvas');
  const sampleContext = sample.getContext('2d', {willReadFrequently: true});
  const ramp = '    ``..--^^~~<>??123456789%%&&@@';
  let assets, options, playlist = [], index = 0, sourceEpoch = 0;
  let suspended = false, paused = false, frameHandle = null, fallback = false;
  let lastRender = -Infinity, lastTime = 0, frames = 0, loops = 0, gl, program, texture, locations;
  let glyphColors = [], backgroundColors = [], renderFailure = '';
  const failedSources = new Set();

  function notify(kind, detail = {}) {
    console.info('VEXA_VIDEO:' + JSON.stringify({kind, ...detail}));
  }
  function rgb(hex) {
    return [1, 3, 5].map(start => parseInt(hex.slice(start, start + 2), 16));
  }
  function rebuildColors() {
    const a = options.ascii, threshold=a.threshold/100, first = rgb(a.color), second = rgb(a.color2);
    const background = rgb(a.background);
    glyphColors = Array.from({length:256}, (_, value) => {
      const t = Math.max(0, Math.min(1, (value / 255 - threshold) / (1 - threshold)));
      return `rgb(${first.map((v, i) => Math.round(v + (second[i] - v) * t)).join(',')})`;
    });
    // The selected background's hue is retained; saturation and luminance are adjustable.
    const max = Math.max(...background), min = Math.min(...background), delta = max - min;
    let hue = delta === 0 ? 0 : max === background[0] ?
      ((background[1] - background[2]) / delta) % 6 : max === background[1] ?
      (background[2] - background[0]) / delta + 2 : (background[0] - background[1]) / delta + 4;
    hue = (hue * 60 + 360) % 360;
    backgroundColors = Array.from({length:256}, (_, value) =>
      `hsl(${hue} ${a.saturation}% ${Math.max(threshold / 4, (value / 255) ** 2) * 100}%)`);
  }
  function cancelFrame() {
    if (frameHandle !== null) {
      if (fallback) cancelAnimationFrame(frameHandle);
      else video.cancelVideoFrameCallback(frameHandle);
      frameHandle = null;
    }
  }
  function active() { return playlist.length && !suspended && !paused; }
  function schedule() {
    if (!active() || frameHandle !== null) return;
    fallback = !video.requestVideoFrameCallback;
    frameHandle = fallback ? requestAnimationFrame(onFrame) : video.requestVideoFrameCallback(onFrame);
  }
  function onFrame(now) {
    frameHandle = null;
    if (!active()) return;
    if (video.loop && video.currentTime + .1 < lastTime) loops++;
    lastTime = video.currentTime;
    if (now - lastRender >= 1000 / options.fps - 1) {
      draw(now);
      lastRender = now;
      frames++;
    }
    schedule();
  }
  function resume() {
    if (!active()) { video.pause(); cancelFrame(); return; }
    const epoch = sourceEpoch;
    video.play().then(schedule).catch(error => {
      if (epoch === sourceEpoch) notify('error', {message: 'Video playback: ' + error.message});
    });
  }
  function select(next) {
    cancelFrame();
    video.pause();
    index = (next + playlist.length) % playlist.length;
    sourceEpoch++;
    lastRender = -Infinity;
    lastTime = 0;
    video.muted = true;
    video.volume = 0;
    video.loop = playlist.length === 1;
    video.src = playlist[index].url;
    video.load();
    notify('source', {index, name: playlist[index].name});
  }
  function advance() {
    for (let step=1; step<=playlist.length; step++) {
      const next=(index+step)%playlist.length;
      if (!failedSources.has(next)) { select(next); return; }
    }
    video.pause(); cancelFrame();
  }
  function dimensions() {
    const w = Math.max(1, document.documentElement.clientWidth);
    const h = Math.max(1, document.documentElement.clientHeight);
    const scale = Math.min(1, 960 / w, 1440 / h);
    return [Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale))];
  }
  function contentRect(width,height) {
    if (options.fit==='fill') return [0,0,width,height];
    const ratio=video.videoWidth/video.videoHeight, aspect=width/height;
    const dw=ratio>aspect ? width : width*ratio/aspect;
    const dh=ratio>aspect ? height*aspect/ratio : height;
    return [(width-dw)/2,(height-dh)/2,dw,dh];
  }
  function drawSource(ctx, width, height, displayAspect=width/height) {
    ctx.fillStyle = options.mode==='ascii' ? options.ascii.background : '#080c10';
    ctx.fillRect(0, 0, width, height);
    const vw = video.videoWidth, vh = video.videoHeight;
    const ratio=vw/vh;
    if (options.fit==='fill') {
      const sw=ratio>displayAspect ? vh*displayAspect : vw;
      const sh=ratio>displayAspect ? vh : vw/displayAspect;
      ctx.drawImage(video,(vw-sw)/2,(vh-sh)/2,sw,sh,0,0,width,height);
    } else {
      const dw=ratio>displayAspect ? width : width*ratio/displayAspect;
      const dh=ratio>displayAspect ? height*displayAspect/ratio : height;
      ctx.drawImage(video,(width-dw)/2,(height-dh)/2,dw,dh);
    }
  }
  function initGL() {
    gl = pixel.getContext('webgl', {alpha:false, antialias:false, depth:false});
    if (!gl) throw new Error('WebGL unavailable. Choose Normal or ASCII.');
    const compile = (type, text) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, text);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const error = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw new Error(error);
      }
      return shader;
    };
    program = gl.createProgram();
    const vertex = compile(gl.VERTEX_SHADER, `attribute vec2 position; varying vec2 vTexCoord;
      void main() { gl_Position=vec4(position,0.,1.); vTexCoord=(position+1.)*.5; vTexCoord.y=1.-vTexCoord.y; }`);
    const fragment = compile(gl.FRAGMENT_SHADER, assets.shader);
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    gl.deleteShader(vertex); gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(pos); gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
    texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture);
    for (const axis of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,axis,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    locations = Object.fromEntries(['resolution','pixelSize','ditherFactor','edgeThreshold',
      'edgeIntensity','edgeColor','palette[0]','paletteCount'].map(key => [key,gl.getUniformLocation(program,key)]));
  }
  function drawPixel(width, height) {
    if (!program) initGL();
    if (gl.isContextLost()) return;
    if (pixel.width !== width || pixel.height !== height) { pixel.width=width; pixel.height=height; }
    if (sample.width !== width || sample.height !== height) { sample.width=width; sample.height=height; }
    drawSource(sampleContext, width, height);
    gl.viewport(0,0,width,height); gl.useProgram(program); gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,sample);
    const p = options.pixel, colors = assets.palettes[p.palette];
    gl.uniform2f(locations.resolution,width,height);
    gl.uniform1f(locations.pixelSize,p.size); gl.uniform1f(locations.ditherFactor,p.dither);
    gl.uniform1f(locations.edgeThreshold,p.edgeThreshold); gl.uniform1f(locations.edgeIntensity,p.edgeIntensity);
    gl.uniform3fv(locations.edgeColor,rgb(p.edgeColor).map(v=>v/255));
    const packed = new Float32Array(16*3); packed.set(colors.flat());
    gl.uniform3fv(locations['palette[0]'],packed); gl.uniform1i(locations.paletteCount,colors.length);
    const [x,y,w,h]=contentRect(width,height);
    const bg=rgb('#080c10').map(value=>value/255);
    gl.clearColor(...bg,1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(Math.ceil(x),Math.ceil(y),Math.floor(w),Math.floor(h));
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    gl.disable(gl.SCISSOR_TEST);
  }
  function drawASCII(width, height, now) {
    if (ascii.width !== width || ascii.height !== height) { ascii.width=width; ascii.height=height; }
    const a = options.ascii, cellW = Math.ceil(Math.min(width,height)/a.resolution), cellH = cellW;
    const columns = Math.ceil(width/cellW), threshold=a.threshold/100, randomness=a.randomness/100;
    const rows = Math.max(1, Math.ceil(height / cellH));
    if (sample.width !== columns || sample.height !== rows) { sample.width=columns; sample.height=rows; }
    // Match upstream resolution on the shorter axis; sample once per square cell.
    drawSource(sampleContext, columns, rows, width / height);
    const pixels = sampleContext.getImageData(0,0,columns,rows).data;
    if (a.effectWidth < 1) drawSource(context,width,height);
    else { context.fillStyle=a.background; context.fillRect(0,0,width,height); }
    context.fillStyle=a.background; context.fillRect(0,0,width*a.effectWidth,height);
    const baseFontSize=cellW/.65;
    const fontSize = Math.min(baseFontSize*3,baseFontSize*a.fontSizeFactor/3), text = Array.from(a.text || 'wavesand');
    const [left,top,contentWidth,contentHeight]=contentRect(width,height);
    context.font=`${fontSize}px monospace`; context.textBaseline='middle'; context.textAlign='center';
    for (let row=0; row<rows; row++) for (let col=0; col<Math.ceil(columns*a.effectWidth); col++) {
      const cx=(col+.5)*cellW, cy=(row+.5)*cellH;
      if (cx<left || cx>left+contentWidth || cy<top || cy>top+contentHeight) continue;
      const offset=(row*columns+col)*4;
      let value=.299*pixels[offset]+.587*pixels[offset+1]+.114*pixels[offset+2];
      if (a.invert) value=255-value;
      value=Math.max(0,Math.min(255,Math.round(value)));
      const x=col*cellW,y=row*cellH;
      if (a.gradient) { context.fillStyle=backgroundColors[value]; context.fillRect(x,y,cellW+1,cellH+1); }
      if (value/255<=threshold || fontSize<=0) continue;
      let character=a.textMode === 'user' ? text[(row*columns+col)%text.length] : ramp[Math.floor(value/255*(ramp.length-1))];
      if (randomness && Math.random()<randomness*.02) character=ramp[Math.floor(Math.random()*ramp.length)];
      if (randomness && Math.sin(now/600+col*.6)*randomness>.9) continue;
      if (a.textMode==='user') context.font=`${Math.max(1,fontSize*value/255)}px monospace`;
      context.fillStyle=glyphColors[value]; context.fillText(character,x+cellW/2,y+cellH/2);
    }
  }
  function draw(now=performance.now()) {
    if (!options || video.readyState<2) return;
    try {
      const [width,height]=dimensions();
      if (options.mode==='pixel') drawPixel(width,height);
      else if (options.mode==='ascii') drawASCII(width,height,now);
      renderFailure='';
    } catch (error) {
      if (error.message!==renderFailure) notify('error',{message:error.message});
      renderFailure=error.message;
    }
  }
  video.addEventListener('loadeddata',()=> { draw(); resume(); });
  video.addEventListener('ended',()=> { loops++; if (playlist.length>1) advance(); });
  video.addEventListener('error',()=> {
    if (!playlist.length) return;
    failedSources.add(index);
    notify('error',{message:`Cannot decode ${playlist[index].name}. Try H.264 MP4 or VP8/VP9 WebM.`,index});
    advance();
  });
  pixel.addEventListener('webglcontextlost',event=> { event.preventDefault(); notify('error',{message:'Pixel renderer GPU context lost; waiting for recovery.'}); });
  pixel.addEventListener('webglcontextrestored',()=> { program=null; draw(); notify('recovered'); });
  new ResizeObserver(()=>draw()).observe(document.documentElement);
  window.VexaVideo = {
    initialize(data) { assets=data; notify('ready'); },
    configure(data) {
      options=data; rebuildColors();
      video.style.objectFit=data.fit==='fill'?'cover':'contain';
      video.style.display=data.mode==='normal'?'block':'none';
      pixel.style.display=data.mode==='pixel'?'block':'none';
      ascii.style.display=data.mode==='ascii'?'block':'none';
      draw();
    },
    setPlaylist(items,selected=0,preserve=false) {
      const previousURL=playlist[index]?.url;
      playlist=items;
      failedSources.clear();
      if (items.length && preserve && items[selected]?.url===previousURL) {
        index=selected; video.loop=items.length===1;
        notify('source',{index,name:items[index].name});
      } else if (items.length) select(selected);
      else { cancelFrame(); video.pause(); video.removeAttribute('src'); video.load();
        for (const element of [video,pixel,ascii]) element.style.display='none'; }
    },
    next() { if (playlist.length) advance(); },
    setPaused(value) { paused=value; resume(); },
    setSuspended(value) { suspended=value; resume(); },
    state() { return {mode:options?.mode, index, frames, loops, time:video.currentTime,
      duration:video.duration, muted:video.muted, volume:video.volume, paused:video.paused,
      ready:video.readyState, error:renderFailure, width:video.videoWidth,height:video.videoHeight,
      pixelSize:options?.pixel.size, resolution:options?.ascii.resolution, fontSizeFactor:options?.ascii.fontSizeFactor,
      asciiOptions:options?.ascii}; },
    shutdown() { suspended=true; cancelFrame(); video.pause(); video.removeAttribute('src'); video.load(); }
  };
})();
