import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ReconstructionService {
  readonly baseUrl = 'http://localhost:8000';
  readonly httpOptions = {
    responseType: 'blob' as 'json',
  };

  constructor(private httpClient: HttpClient) {}

  getPrediction(photo: string): Observable<Blob> {
    const formData = new FormData();
    formData.append('image_file', photo);
    return this.httpClient.post<Blob>(
      `${this.baseUrl}/predict`,
      formData,
      this.httpOptions
    );
  }
}
