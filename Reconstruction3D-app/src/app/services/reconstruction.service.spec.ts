import { TestBed } from '@angular/core/testing';

import { ReconstructionService } from './reconstruction.service';

describe('ReconstructionService', () => {
  let service: ReconstructionService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ReconstructionService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
