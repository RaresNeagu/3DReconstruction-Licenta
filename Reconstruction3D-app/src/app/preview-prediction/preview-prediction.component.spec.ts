import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreviewPredictionComponent } from './preview-prediction.component';

describe('PreviewPredictionComponent', () => {
  let component: PreviewPredictionComponent;
  let fixture: ComponentFixture<PreviewPredictionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PreviewPredictionComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(PreviewPredictionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
