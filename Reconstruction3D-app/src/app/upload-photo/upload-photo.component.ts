import { Component, OnInit } from '@angular/core';
import { ReconstructionService } from '../services/reconstruction.service';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { ThemePalette } from '@angular/material/core';

@Component({
  selector: 'app-upload-photo',
  templateUrl: './upload-photo.component.html',
  styleUrls: ['./upload-photo.component.scss'],
})
export class UploadPhotoComponent implements OnInit {
  obj: Blob;
  objUrl: SafeUrl;
  showSpinner: boolean = false;
  color: ThemePalette = 'primary';
  constructor(
    private reconstructionService: ReconstructionService,
    private domSanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {}

  onChangePhoto(event: any) {
    if (event.target.files.length > 0) {
      const file = event.target.files[0];
      if (file.type == 'image/png' || file.type == 'image/jpeg') {
        this.showSpinner = true;
        this.reconstructionService.getPrediction(file).subscribe((obj) => {
          this.objUrl = this.domSanitizer.bypassSecurityTrustUrl(
            URL.createObjectURL(obj)
          );
          this.obj = obj;
          this.showSpinner = false;
          event.target.value = '';
        });
      } else {
        alert('Only PNG and JPEG are accepted.');
      }
    }
  }
}
