import re
import sys

try:
    with open('dashboard/js/pinn_canvas_backup.js', 'r', encoding='utf-8') as f:
        code = f.read()

    # Replace initFloodCanvas
    code = re.sub(r'async function initFloodCanvas\(\).*?(?=\n// ── LOAD REGION)', '''async function initFloodCanvas() {
  const container = document.getElementById('pinn-leaflet-map');
  if (!container) return;

  container.innerHTML = '';
  PINN_SIM.scene = new THREE.Scene();
  PINN_SIM.scene.background = new THREE.Color(0x060b14);
  PINN_SIM.scene.fog = new THREE.FogExp2(0x060b14, 0.0015);

  const width = container.clientWidth;
  const height = container.clientHeight;
  PINN_SIM.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000);
  PINN_SIM.camera.position.set(-300, 600, 800);

  PINN_SIM.renderer = new THREE.WebGLRenderer({ antialias: true });
  PINN_SIM.renderer.setSize(width, height);
  PINN_SIM.renderer.setPixelRatio(window.devicePixelRatio);
  PINN_SIM.renderer.shadowMap.enabled = true;
  container.appendChild(PINN_SIM.renderer.domElement);

  PINN_SIM.controls = new THREE.OrbitControls(PINN_SIM.camera, PINN_SIM.renderer.domElement);
  PINN_SIM.controls.enableDamping = true;
  PINN_SIM.controls.dampingFactor = 0.05;
  PINN_SIM.controls.maxPolarAngle = Math.PI / 2 - 0.05;

  const ambientLight = new THREE.AmbientLight(0x404040, 1.2);
  PINN_SIM.scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(500, 1000, 200);
  dirLight.castShadow = true;
  PINN_SIM.scene.add(dirLight);

  generateTerrain();

  window.addEventListener('resize', () => {
    if (!PINN_SIM.camera || !PINN_SIM.renderer) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    PINN_SIM.camera.aspect = w / h;
    PINN_SIM.camera.updateProjectionMatrix();
    PINN_SIM.renderer.setSize(w, h);
  }, false);

  animate();

  await loadPINNRegion(PINN_SIM.activeRegion);
  checkAutoRiskState();
  if (!PINN_SIM.pollTimer) {
    PINN_SIM.pollTimer = setInterval(checkAutoRiskState, 1500);
  }
}

function generateTerrain() {
  const geometry = new THREE.PlaneGeometry(3000, 3000, 250, 250);
  geometry.rotateX(-Math.PI / 2);

  const pos = geometry.attributes.position;
  const colors = [];
  const colorObj = new THREE.Color();

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);

    const distFromCenter = Math.abs(x);
    let height = (distFromCenter / 1500) * 800;

    height += Math.sin(x * 0.01) * Math.cos(z * 0.01) * 200;
    height += Math.sin(x * 0.03) * Math.cos(z * 0.03) * 80;
    height += Math.sin(x * 0.1) * Math.cos(z * 0.1) * 15;
    
    if (distFromCenter < 200) {
      height -= (200 - distFromCenter) * 0.8; 
    }
    
    // Add slope towards z end to make river flow downwards
    height += (z + 1500) * 0.1;

    height = Math.max(0, height);
    pos.setY(i, height);

    if (height < 80) {
      colorObj.setHex(0x1a4314); 
    } else if (height < 250) {
      colorObj.setHex(0x354b2a); 
    } else if (height < 500) {
      colorObj.setHex(0x6e5c47); 
    } else if (height < 800) {
      colorObj.setHex(0x8b8b8b);
    } else {
      colorObj.setHex(0xffffff); 
    }
    colors.push(colorObj.r, colorObj.g, colorObj.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.9,
    flatShading: true
  });

  PINN_SIM.terrainMesh = new THREE.Mesh(geometry, material);
  PINN_SIM.terrainMesh.receiveShadow = true;
  PINN_SIM.terrainMesh.castShadow = true;
  PINN_SIM.scene.add(PINN_SIM.terrainMesh);

  const waterGeo = new THREE.PlaneGeometry(350, 3000, 20, 100);
  waterGeo.rotateX(-Math.PI / 2);
  
  // slope water slightly
  const wpos = waterGeo.attributes.position;
  for(let i=0; i<wpos.count; i++) {
     wpos.setY(i, (wpos.getZ(i) + 1500)*0.1);
  }
  
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4,
    transparent: true,
    opacity: 0.85,
    roughness: 0.1,
    metalness: 0.8
  });
  PINN_SIM.waterMesh = new THREE.Mesh(waterGeo, waterMat);
  PINN_SIM.waterMesh.position.y = -200; 
  PINN_SIM.scene.add(PINN_SIM.waterMesh);
  
  PINN_SIM.waterTargetY = -200;
  PINN_SIM.waterCurrentY = -200;
}

function animate() {
  PINN_SIM.animationId = requestAnimationFrame(animate);
  PINN_SIM.controls.update();

  if (PINN_SIM.waterMesh) {
    PINN_SIM.waterCurrentY += (PINN_SIM.waterTargetY - PINN_SIM.waterCurrentY) * 0.05;
    PINN_SIM.waterMesh.position.y = PINN_SIM.waterCurrentY;
  }

  updateTownLabels();
  PINN_SIM.renderer.render(PINN_SIM.scene, PINN_SIM.camera);
}

function renderTownsForRegion(regionKey) {
  PINN_SIM.townMarkers.forEach(m => {
    if (m.element && m.element.parentNode) {
      m.element.parentNode.removeChild(m.element);
    }
  });
  PINN_SIM.townMarkers = [];

  const anchors = BASIN_ANCHORS[regionKey];
  if (!anchors) return;
  
  const container = document.getElementById('pinn-leaflet-map');

  anchors.forEach(a => {
    const z = (a.y * 2400) - 1200;
    const x = (Math.random() - 0.5) * 100; 
    const y = 150 + ((z + 1500) * 0.1); 

    const el = document.createElement('div');
    el.className = 'pinn-3d-label';
    el.innerHTML = '<span style="color:#ef4444; font-size:16px;">📍</span> ' + a.name + ' (' + a.elev + 'm)';
    el.style.position = 'absolute';
    el.style.color = 'white';
    el.style.fontWeight = '600';
    el.style.fontFamily = 'var(--font-sans)';
    el.style.textShadow = '0 2px 4px rgba(0,0,0,0.9)';
    el.style.pointerEvents = 'none';
    el.style.fontSize = '12px';
    el.style.transform = 'translate(-50%, -100%)';
    el.style.background = 'rgba(0,0,0,0.3)';
    el.style.padding = '2px 6px';
    el.style.borderRadius = '4px';
    el.style.border = '1px solid rgba(255,255,255,0.1)';
    container.appendChild(el);

    PINN_SIM.townMarkers.push({
      element: el,
      pos3D: new THREE.Vector3(x, y, z)
    });
  });
}

function updateTownLabels() {
  if (!PINN_SIM.camera || !PINN_SIM.renderer) return;
  
  const widthHalf = PINN_SIM.renderer.domElement.clientWidth / 2;
  const heightHalf = PINN_SIM.renderer.domElement.clientHeight / 2;

  PINN_SIM.townMarkers.forEach(m => {
    const pos = m.pos3D.clone();
    pos.project(PINN_SIM.camera);

    if (pos.z > 1 || pos.z < -1) {
      m.element.style.display = 'none';
      return;
    }
    
    if (pos.x < -1 || pos.x > 1 || pos.y < -1 || pos.y > 1) {
      m.element.style.display = 'none';
      return;
    }

    m.element.style.display = 'block';
    const x = (pos.x * widthHalf) + widthHalf;
    const y = -(pos.y * heightHalf) + heightHalf;
    m.element.style.left = x + 'px';
    m.element.style.top = y + 'px';
  });
}
''', code, flags=re.DOTALL)

    # Replace applyPINNState logic that manipulates Leaflet
    code = re.sub(r'    if \(PINN_SIM\.floodGroup && PINN_SIM\.pinnMap\) \{.*?\n      PINN_SIM\.floodGroup = null;\n    \}', '''    PINN_SIM.waterTargetY = -200;''', code, flags=re.DOTALL)

    # Remove loadPINNRegion leaflet calls (fitBounds, addLayer)
    code = re.sub(r'  PINN_SIM\.pinnMap\.fitBounds.*?townsEl\.innerHTML =.*?\}', '''  renderTownsForRegion(normalizedKey);
  const townsEl = document.getElementById('pinn-towns-list');
  if (townsEl) {
    const names = data.towns_affected.map(t => t.name).join(" &bull; ");
    townsEl.innerHTML = '<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: var(--text-main); font-weight:600;">' + names + '</span>';
  }
''', code, flags=re.DOTALL)

    # Replace renderFloodDepthOverlay leaflet logic with 3D trigger
    code = re.sub(r'function renderFloodDepthOverlay.*?\}', '''function renderFloodDepthOverlay(grid, peakDepth, anchors) {
  // In 3D, we just raise the water mesh!
  PINN_SIM.waterTargetY = 80; 
}''', code, flags=re.DOTALL)

    # Clean up initial object
    code = code.replace('pinnMap: null,', 'scene: null, camera: null, renderer: null, controls: null, terrainMesh: null, waterMesh: null, townMarkers: [], waterTargetY: -200, waterCurrentY: -200,')

    # Remove the DOMContentLoaded event from this file since it causes a duplicate map
    # Or just write it back
    with open('dashboard/js/pinn_canvas.js', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Successfully rewrote pinn_canvas.js")
except Exception as e:
    print("Error:", e)
    sys.exit(1)
