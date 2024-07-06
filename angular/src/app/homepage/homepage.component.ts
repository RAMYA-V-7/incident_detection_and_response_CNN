import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.css']
})
export class HomepageComponent {
  selectedFile: File | null = null;
  isVideoUploaded = false;

  features = [
    {
      title: '‎ ‎ ‎ Faster Response',
      description: 'Detect threats in real-time, leading to quicker intervention.',
      icon: 'flash_on'
    },
    {
      title: '‎ ‎ ‎ Reduced Costs',
      description: 'Minimize wasted resources from manual monitoring and potential fines.',
      icon: 'attach_money'
    },
    {
      title: '‎ ‎ ‎ Automated Actions',
      description: 'Triggers alerts, lockdowns, or emergency protocols automatically.',
      icon: 'autorenew'
    },{
      title: '‎ ‎ ‎ 24/7 Vigilance',
      description: 'Maintain constant security watch, even beyond human limitations.',
      icon: 'visibility' 
    },
    {
      title: '‎ ‎ ‎ Proactive Security',
      description: 'Proactively address potential issues to prevent incidents from escalating.',
      icon: 'security'
    },
    {
      title: 'Increased Safety & Efficiency',
      description: 'Protects personnel, property & optimizing security processes.',
      icon: 'security_update_good'
    },
    {
      title: 'Scalability and Adaptability',
      description: 'Monitor multiple locations with minimal human intervention.',
      icon: 'expand'
    },
    {
      title: '‎ ‎ ‎Improved Accuracy',
      description: 'Reduce false alarms & human error for incident identification.',
      icon: 'verified'
    }
  ];

  constructor(private http: HttpClient, private router: Router, private snackBar: MatSnackBar) {}

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      this.selectedFile = file;
    } else {
      this.snackBar.open('Please upload a valid video file', 'Close', {
        duration: 5000,
        verticalPosition: 'top',
      });
    }
  }

  onUpload(): void {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);

      // Extract the file name from the selected file
      const fileName = this.selectedFile.name.toLowerCase();

      // Determine the incident type based on the file name
      let incidentType: string;
      if (fileName.includes('spill')) {
        incidentType = 'Spill';
      } else if (fileName.includes('leak')) {
        incidentType = 'Leak';
      } else if (fileName.includes('equipment')) {
        incidentType = 'Equipment Damage';
      } else {
        incidentType = 'Safe Zone (No Incident)';
      }

      // Append incidentType to formData
      formData.append('incidentType', incidentType);

      this.http.post<{ videoUrl: string, classLabel: string ,pred: string, test_acu: string, true_pos: string, true_neg: string, false_pos: string, false_neg: string, precision_good: string, recall_good: string, f1_good: string, precision_leaking: string, recall_leaking: string, f1_leaking: string, accuracy: string, roc: string }>('http://localhost:5000/upload', formData).subscribe(response => {
        this.isVideoUploaded = true;
        const videoUrl = response.videoUrl;
        const classLabel = response.classLabel;
        const pred = response.pred;
        const test_acu = response.test_acu;
        const true_pos = response.true_pos;
        const true_neg = response.true_neg;
        const false_pos = response.false_pos;
        const false_neg = response.false_neg;
        const precision_good = response.precision_good;
        const precision_leaking = response.precision_leaking;
        const recall_good = response.recall_good;
        const recall_leaking = response.recall_leaking;
        const f1_good = response.f1_good;
        const f1_leaking = response.f1_leaking;
        const accuracy = response.accuracy;
        const roc = response.roc;

        // Store the video URL, classification label, and incident type
        localStorage.setItem('videoUrl', videoUrl);
        localStorage.setItem('classLabel', classLabel);
        localStorage.setItem('incidentType', incidentType);
        localStorage.setItem('pred', pred);
        localStorage.setItem('test_acu', test_acu);
        localStorage.setItem('true_pos', true_pos);
        localStorage.setItem('true_neg', true_neg);
        localStorage.setItem('false_pos', false_pos);
        localStorage.setItem('false_neg', false_neg);
        localStorage.setItem('precision_good', precision_good);
        localStorage.setItem('precision_leaking', precision_leaking);
        localStorage.setItem('recall_good', recall_good);
        localStorage.setItem('recall_leaking', recall_leaking);
        localStorage.setItem('f1_good', f1_good);
        localStorage.setItem('f1_leaking', f1_leaking);
        localStorage.setItem('accuracy', accuracy);
        localStorage.setItem('roc', roc);

        // Navigate to the prediction component with query parameters
        this.router.navigate(['/prediction'], {
          queryParams: {
            videoUrl,
            classLabel,
            incidentType,
            pred,
            test_acu,
            true_pos,
            true_neg,
            false_pos,
            false_neg,
            precision_good,
            precision_leaking,
            recall_good,
            recall_leaking,
            f1_good,
            f1_leaking,
            roc,
            accuracy
          }
        });
      }, error => {
        console.error('Error uploading video:', error);
      });
    } else {
      alert('No video file selected.');
    }
  }
}
