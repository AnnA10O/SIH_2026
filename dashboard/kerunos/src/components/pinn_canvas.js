import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export const BASIN_ANCHORS = [
  {
    id: "kedarnath",
    name: "Kedarnath Temple",
    elevation: "3,583 m",
    coords: { lat: 30.7346, lon: 79.0669 },
    pos: { x: 0, y: 720, z: -1100 },
    camPos: { x: 450, y: 880, z: -700 },
    desc: "Glacial Source & Ridge Peak"
  },
  {
    id: "rambara",
    name: "Rambara Gorge",
    elevation: "2,740 m",
    coords: { lat: 30.6820, lon: 79.0550 },
    pos: { x: 0, y: 460, z: -300 },
    camPos: { x: 400, y: 620, z: 50 },
    desc: "Critical Inundation Funnel"
  },
  {
    id: "gaurikund",
    name: "Gaurikund Base",
    elevation: "1,980 m",
    coords: { lat: 30.6515, lon: 79.0252 },
    pos: { x: 0, y: 310, z: 400 },
    camPos: { x: 380, y: 480, z: 750 },
    desc: "Pilgrim Base Station"
  },
  {
    id: "sonprayag",
    name: "Sonprayag Confluence",
    elevation: "1,820 m",
    coords: { lat: 30.6300, lon: 79.0200 },
    pos: { x: 0, y: 210, z: 1100 },
    camPos: { x: 350, y: 390, z: 1450 },
    desc: "Mandakini River Outlet"
  }
];

export class PinnFloodCanvasEngine {
  constructor(container, options = {}) {
    this.container = container;
    this.onLabelUpdate = options.onLabelUpdate || null;
    this.onWaypointSelect = options.onWaypointSelect || null;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    this.terrainMesh = null;
    this.waterMesh = null;
    this.rainParticles = null;
    this.flowParticles = null;

    this.activeWaypointIndex = 0;
    this.isFlightActive = false;
    this.flightProgress = 0;
    this.flightStartCam = new THREE.Vector3();
    this.flightEndCam = new THREE.Vector3();
    this.flightStartTarget = new THREE.Vector3();
    this.flightEndTarget = new THREE.Vector3();

    this.waterHeightOffset = -60;
    this.targetWaterOffset = -60;
    this.rainfallRate = 0;
    this.currentTheme = options.theme || 'dark';

    this.animId = null;
    this.boundKeyDown = this.handleKeyDown.bind(this);

    this.initFloodCanvas();
  }

  initFloodCanvas() {
    if (typeof THREE === 'undefined') {
      console.error('[PINN] Three.js failed to load.');
      return;
    }
    if (typeof OrbitControls === 'undefined') {
      console.error('[PINN] OrbitControls failed to load.');
      return;
    }

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    if (width <= 0 || height <= 0) {
      requestAnimationFrame(() => this.initFloodCanvas());
      return;
    }

    console.log('[PINN] Three.js loaded');
    console.log('[PINN] Initializing 3D engine');
    console.log(`[PINN] Container size: ${width} x ${height}`);

    // 1. Scene setup
    this.scene = new THREE.Scene();
    
    // 2. Camera setup
    this.camera = new THREE.PerspectiveCamera(45, width / height, 10, 8000);
    const initialAnchor = BASIN_ANCHORS[0];
    this.camera.position.set(initialAnchor.camPos.x, initialAnchor.camPos.y, initialAnchor.camPos.z);

    // 3. Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Clear previous canvas if exists
    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2.05; // Don't go below ground
    this.controls.minDistance = 100;
    this.controls.maxDistance = 4500;
    this.controls.target.set(initialAnchor.pos.x, initialAnchor.pos.y, initialAnchor.pos.z);
    this.controls.update();

    // 5. Lighting
    this.hemiLight = new THREE.HemisphereLight(0xf8fafc, 0x1e293b, 0.85);
    this.scene.add(this.hemiLight);

    this.dirLight = new THREE.DirectionalLight(0x38bdf8, 1.25);
    this.dirLight.position.set(600, 1200, -800);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.width = 2048;
    this.dirLight.shadow.mapSize.height = 2048;
    this.dirLight.shadow.camera.near = 100;
    this.dirLight.shadow.camera.far = 4000;
    const d = 1500;
    this.dirLight.shadow.camera.left = -d;
    this.dirLight.shadow.camera.right = d;
    this.dirLight.shadow.camera.top = d;
    this.dirLight.shadow.camera.bottom = -d;
    this.scene.add(this.dirLight);

    this.ambientLight = new THREE.AmbientLight(0x0f172a, 0.4);
    this.scene.add(this.ambientLight);

    // 6. Generate Terrain & Water
    this.generateTerrain();
    this.createWaterMesh();
    this.createRainfallParticles();
    this.createFlowParticles();

    // Set Theme Colors
    this.setTheme(this.currentTheme);

    // 7. Event listeners
    window.addEventListener('keydown', this.boundKeyDown, true);
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.container);

    console.log('[PINN] Terrain generated');
    console.log('[PINN] Water generated');
    console.log('[PINN] PINN engine ready');

    // 8. Start loop
    this.animate();
  }

  generateTerrain() {
    const size = 3000;
    const segments = 120;
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    const colors = [];

    const cGreen = new THREE.Color("#15803d");
    const cMid = new THREE.Color("#78350f");
    const cGrey = new THREE.Color("#64748b");
    const cSnow = new THREE.Color("#f8fafc");

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);

      // Downstream slope along Z-axis (Kedarnath z=-1500 to Sonprayag z=+1500)
      const downstreamSlope = (z + 1500) * 0.25;

      // Parabolic Valley shape along X-axis
      const absX = Math.abs(x);
      let valleyFactor = Math.pow(absX / 1400, 2.2) * 680;
      if (absX < 180) valleyFactor *= 0.15; // Flatter central riverbed

      // Extreme outer crust drop for solid mountain boundary look
      if (absX > 1350) {
        valleyFactor += Math.pow((absX - 1350) / 150, 2) * 300;
      }

      // Fractal multi-layer noise for steep Himalayan ridges
      const noise1 = Math.sin(x * 0.004) * Math.cos(z * 0.004) * 140;
      const noise2 = Math.sin(x * 0.012 + z * 0.01) * 55;
      const noise3 = Math.cos(x * 0.025 - z * 0.02) * 25;
      const totalNoise = (absX < 150) ? (noise1 + noise2 + noise3) * 0.3 : (noise1 + noise2 + noise3);

      const y = valleyFactor + totalNoise + downstreamSlope;
      pos.setY(i, y);

      // Vertex color based on elevation Y
      const vColor = new THREE.Color();
      if (y < 260) {
        vColor.lerpColors(cGreen, cMid, (y - 50) / 210);
      } else if (y < 520) {
        vColor.lerpColors(cMid, cGrey, (y - 260) / 260);
      } else {
        vColor.lerpColors(cGrey, cSnow, Math.min(1.0, (y - 520) / 300));
      }
      colors.push(vColor.r, vColor.g, vColor.b);
    }

    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.15,
      flatShading: true
    });

    this.terrainMesh = new THREE.Mesh(geometry, material);
    this.terrainMesh.receiveShadow = true;
    this.terrainMesh.castShadow = true;
    this.scene.add(this.terrainMesh);
  }

  createWaterMesh() {
    const width = 550;
    const length = 3000;
    const geometry = new THREE.PlaneGeometry(width, length, 30, 120);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const z = pos.getZ(i);
      const downstreamSlope = (z + 1500) * 0.25;
      pos.setY(i, downstreamSlope + this.waterHeightOffset);
    }

    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      transparent: true,
      opacity: 0.75,
      roughness: 0.1,
      metalness: 0.7,
      flatShading: false
    });

    this.waterMesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.waterMesh);
  }

  setTheme(theme) {
    this.currentTheme = theme;
    if (!this.scene) return;

    if (theme === 'light') {
      this.scene.background = new THREE.Color('#e2e8f0');
      this.scene.fog = new THREE.FogExp2('#e2e8f0', 0.00035);
      if (this.ambientLight) this.ambientLight.color.set('#f8fafc');
      if (this.ambientLight) this.ambientLight.intensity = 0.7;
      if (this.hemiLight) this.hemiLight.color.set('#ffffff');
      if (this.dirLight) this.dirLight.intensity = 1.5;
    } else {
      this.scene.background = new THREE.Color('#0a0f1c');
      this.scene.fog = new THREE.FogExp2('#0a0f1c', 0.00045);
      if (this.ambientLight) this.ambientLight.color.set('#0f172a');
      if (this.ambientLight) this.ambientLight.intensity = 0.4;
      if (this.hemiLight) this.hemiLight.color.set('#f8fafc');
      if (this.dirLight) this.dirLight.intensity = 1.25;
    }
  }

  createRainfallParticles() {
    const count = 1200;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 2400;
      positions[i * 3 + 1] = 600 + Math.random() * 800;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 2800;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 3.5,
      transparent: true,
      opacity: 0.05
    });

    this.rainParticles = new THREE.Points(geometry, material);
    this.scene.add(this.rainParticles);
  }

  createFlowParticles() {
    const count = 350;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const z = -1400 + Math.random() * 2800;
      const downstreamSlope = (z + 1500) * 0.25;
      positions[i * 3] = (Math.random() - 0.5) * 160;
      positions[i * 3 + 1] = downstreamSlope + 15;
      positions[i * 3 + 2] = z;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 6.0,
      transparent: true,
      opacity: 0.8
    });

    this.flowParticles = new THREE.Points(geometry, material);
    this.scene.add(this.flowParticles);
  }

  setWaterHeightAndRain(targetHeightOffset, rainRate) {
    this.targetWaterOffset = targetHeightOffset;
    this.rainfallRate = rainRate;
  }

  updateWaterMeshHeight() {
    // Smoothly interpolate current height offset to target
    this.waterHeightOffset += (this.targetWaterOffset - this.waterHeightOffset) * 0.08;

    if (this.waterMesh) {
      const pos = this.waterMesh.geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const z = pos.getZ(i);
        const downstreamSlope = (z + 1500) * 0.25;
        pos.setY(i, downstreamSlope + this.waterHeightOffset);
      }
      this.waterMesh.geometry.attributes.position.needsUpdate = true;
      this.waterMesh.geometry.computeVertexNormals();

      // Opacity scales with flood height
      const targetOpacity = this.waterHeightOffset > 0 ? Math.min(0.88, 0.65 + this.waterHeightOffset * 0.003) : 0.45;
      this.waterMesh.material.opacity = targetOpacity;
    }
  }

  animateRainAndFlow() {
    // Rain Particles
    if (this.rainParticles) {
      const targetOpacity = Math.min(0.75, (this.rainfallRate / 135) * 0.7);
      this.rainParticles.material.opacity += (targetOpacity - this.rainParticles.material.opacity) * 0.1;

      const pos = this.rainParticles.geometry.attributes.position;
      const speed = 12 + (this.rainfallRate / 10);
      for (let i = 0; i < pos.count; i++) {
        let y = pos.getY(i) - speed;
        if (y < 100) y = 1000 + Math.random() * 200;
        pos.setY(i, y);
      }
      this.rainParticles.geometry.attributes.position.needsUpdate = true;
    }

    // Downstream Water Flow Particles
    if (this.flowParticles) {
      const pos = this.flowParticles.geometry.attributes.position;
      const flowSpeed = 6 + (this.waterHeightOffset > 0 ? this.waterHeightOffset * 0.08 : 0);
      for (let i = 0; i < pos.count; i++) {
        let z = pos.getZ(i) + flowSpeed;
        if (z > 1450) z = -1400;
        const downstreamSlope = (z + 1500) * 0.25;
        pos.setZ(i, z);
        pos.setY(i, downstreamSlope + Math.max(12, this.waterHeightOffset + 10));
      }
      this.flowParticles.geometry.attributes.position.needsUpdate = true;
    }
  }

  jumpToTown(index) {
    if (index < 0 || index >= BASIN_ANCHORS.length) return;
    this.activeWaypointIndex = index;
    const anchor = BASIN_ANCHORS[index];

    this.isFlightActive = true;
    this.flightProgress = 0;

    this.flightStartCam.copy(this.camera.position);
    this.flightEndCam.set(anchor.camPos.x, anchor.camPos.y, anchor.camPos.z);

    this.flightStartTarget.copy(this.controls.target);
    this.flightEndTarget.set(anchor.pos.x, anchor.pos.y, anchor.pos.z);

    this.controls.enabled = false;

    if (this.onWaypointSelect) {
      this.onWaypointSelect(index, anchor);
    }
  }

  jumpToNextTown(direction = 1) {
    let nextIdx = (this.activeWaypointIndex + direction) % BASIN_ANCHORS.length;
    if (nextIdx < 0) nextIdx = BASIN_ANCHORS.length - 1;
    this.jumpToTown(nextIdx);
  }

  handleKeyDown(e) {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      this.jumpToNextTown(1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      this.jumpToNextTown(-1);
    }
  }

  updateCameraFlight() {
    if (!this.isFlightActive) return;

    this.flightProgress += 0.035;
    const ease = 0.5 - Math.cos(Math.min(1.0, this.flightProgress) * Math.PI) / 2;

    this.camera.position.lerpVectors(this.flightStartCam, this.flightEndCam, ease);
    this.controls.target.lerpVectors(this.flightStartTarget, this.flightEndTarget, ease);

    if (this.flightProgress >= 1.0) {
      this.isFlightActive = false;
      this.controls.enabled = true;
      this.controls.update();
    }
  }

  updateProjectedLabels() {
    if (!this.onLabelUpdate || !this.camera) return;

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    const projectedLabels = BASIN_ANCHORS.map((anchor, index) => {
      const vec = new THREE.Vector3(
        anchor.pos.x,
        anchor.pos.y + Math.max(25, this.waterHeightOffset + 20),
        anchor.pos.z
      );
      vec.project(this.camera);

      const x = (vec.x * 0.5 + 0.5) * width;
      const y = (-(vec.y * 0.5) + 0.5) * height;
      const isVisible = vec.z < 1.0 && x >= 0 && x <= width && y >= 0 && y <= height;

      return {
        ...anchor,
        index,
        screenX: x,
        screenY: y,
        isVisible,
        isActive: index === this.activeWaypointIndex
      };
    });

    this.onLabelUpdate(projectedLabels);
  }

  animate() {
    this.animId = requestAnimationFrame(() => this.animate());

    this.updateWaterMeshHeight();
    this.animateRainAndFlow();
    this.updateCameraFlight();

    if (!this.isFlightActive && this.controls) {
      this.controls.update();
    }

    this.updateProjectedLabels();
    this.renderer.render(this.scene, this.camera);
  }

  handleResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    window.removeEventListener('keydown', this.boundKeyDown, true);
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.renderer && this.renderer.domElement) {
      this.renderer.domElement.remove();
    }
  }
}

export function initFloodCanvas(container, options) {
  if (!container) {
    console.error('[PINN] flood-canvas-wrap not found');
    return null;
  }
  return new PinnFloodCanvasEngine(container, options);
}
