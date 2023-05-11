import { Component, OnInit } from '@angular/core';
import { ReconstructionService } from '../services/reconstruction.service';

@Component({
  selector: 'app-upload-photo',
  templateUrl: './upload-photo.component.html',
  styleUrls: ['./upload-photo.component.scss'],
})
export class UploadPhotoComponent implements OnInit {
  obj: Blob;
  objUrl: string;
  constructor(private reconstructionService: ReconstructionService) {}

  ngOnInit(): void {}

  onChangePhoto(event: any) {
    if (event.target.files.length > 0) {
      const file = event.target.files[0];
      if (file.type == 'image/png' || file.type == 'image/jpeg') {
        this.reconstructionService.getPrediction(file).subscribe((obj) => {
          this.objUrl = URL.createObjectURL(obj);
          this.obj = obj;
        });
      } else {
        alert('Only PNG and JPEG are accepted.');
      }
    }
  }
}
