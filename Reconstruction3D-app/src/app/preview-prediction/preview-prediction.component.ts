import {
  AfterViewChecked,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnInit,
  ViewChild,
} from '@angular/core';
import * as THREE from 'three';
@Component({
  selector: 'app-preview-prediction',
  templateUrl: './preview-prediction.component.html',
  styleUrls: ['./preview-prediction.component.scss'],
})
export class PreviewPredictionComponent implements OnInit, OnChanges {
  @Input() objFile: Blob;

  constructor() {}

  ngOnInit(): void {}

  ngOnChanges(): void {
    this.parseObjFile(this.objFile);
  }

  createThreeJsBox(prediction: THREE.BufferGeometry): void {
    const canvas = document.getElementById('canvas-box');

    const scene = new THREE.Scene();

    const material = new THREE.MeshBasicMaterial({ color: 0x808080 });

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.5);
    pointLight.position.x = 2;
    pointLight.position.y = 2;
    pointLight.position.z = 2;
    scene.add(pointLight);

    let box;
    if (this.objFile) {
      box = new THREE.Mesh(prediction, material);
    } else {
      box = new THREE.Mesh(new THREE.BoxGeometry(5.5, 5.5, 5.5), material);
    }

    scene.add(box);

    const canvasSizes = {
      width: 700,
      height: 500,
    };

    const camera = new THREE.PerspectiveCamera(
      75,
      canvasSizes.width / canvasSizes.height,
      0.001,
      1000
    );
    camera.position.z = 30;
    scene.add(camera);

    if (!canvas) {
      return;
    }

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
    });
    renderer.setClearColor(0xe232222, 1);
    renderer.setSize(canvasSizes.width, canvasSizes.height);

    const clock = new THREE.Clock();

    const animateGeometry = () => {
      const elapsedTime = clock.getElapsedTime();

      // Update animaiton objects
      box.rotation.y = elapsedTime;

      // Render
      renderer.render(scene, camera);

      // Call tick again on the next frame
      window.requestAnimationFrame(animateGeometry);
    };

    animateGeometry();
  }

  parseObjFile(objBlob: Blob) {
    const geometry = new THREE.BufferGeometry();
    const reader = new FileReader();
    reader.readAsText(objBlob);
    reader.onload = () => {
      const objData = reader.result.toString();
      const positions = [];

      const indices = [];

      const lines = objData.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line[0] === 'v') {
          const values = line.split(' ').map((v) => parseFloat(v));
          positions.push(values[1], values[2], values[3]);
        } else if (line[0] === 'f') {
          const values = line.split(' ').map((v) => parseInt(v));
          indices.push(values[1], values[2], values[3]);
        }
      }

      geometry.setIndex(indices);
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(positions), 3)
      );
      geometry.computeVertexNormals();
      geometry.scale(30.0, 30.0, 30.0);
      this.createThreeJsBox(geometry);
    };
  }
}
