import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { CanvasJSAngularChartsModule } from '@canvasjs/angular-charts';
import { MatSnackBar } from '@angular/material/snack-bar';
@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class PredictionComponent implements OnInit {
  videoUrl: string | null = null;
  classLabel: string | null = null;
  incidentType: string | null = null;
  pred: string | null = null;
  test_acu:number | any;
  true_pos:string | null = null;
  true_neg:string | null = null;
  false_pos:string | null = null;
  false_neg:string | null = null;
  precision_good:string | null = null;
  precision_leaking:string | null = null;
  recall_good:string | null = null;
  recall_leaking:string | null = null;
  f1_good:string | null = null;
  f1_leaking:string | null = null;
  roc:string | null = null;

  
  chartOptions = {
    animationEnabled: true,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    title: {
      text: "Model Performance",
      fontFamily:"Roboto",
       fontWeight: "bold",
      //  color: "rgb(0, 0, 0)"  
       fontSize: 22,
      // color: "rgb(255, 255, 255)",
    },
    data: [{
      type: "doughnut",
      startAngle: 45,
      indexLabel: "{name}: {y}",
      indexLabelPlacement: "outside",
      yValueFormatString: "#,###.##'%'",
      indexLabelFontFamily:"Roboto",
      indexLabelFontWeight: "bold",
      indexLabelFontSize: 14,
      yValueFormatFontFamily:"Roboto",
      yValueFormatFontWeight:"bold",
      yValueFormatFontSize: 10,
      dataPoints: [
            { y: 62.5,name: "Accuracy",toolTipContent: "Correctness of all predictions" },
            { y: 62.5, name: "Classification b/w ✅ and ⚠️", toolTipContent: " How good at separating Incidents from safe situations?" },
            { y: 75, name: "Catches Most ✅", toolTipContent: "How good at catching most safer spots?" },
            { y: 60, name: " Catches ✅ well ", toolTipContent: "How good at capturing safer spots?" },
            { y: 66.7, name: "Most ✅ less misses", toolTipContent: "How well does it catch Safezone without falsely triggering alarms" },
            { y: 57.14, name: "Most ⚠️ less misses", toolTipContent: "How well does it catch Incidents without falsely triggering alarms?" },
            { y: 66.7, name: " Capture ⚠️ well ", toolTipContent: "How good at capturing incidents?" },
            { y: 50, name: "Catches Most ⚠️", toolTipContent: "How good at catching most incidents?" },
      ]
    }]
    
  };
  spills = [
    {
      title: '‎ ‎ Contain it',
      description: 'Use Sandbags, spill berms or dikes to create a temporary pool around the spill.',
      icon: 'block' 
    },
    {
      title: '‎ ‎ ‎ Use absorbent',
      description: 'Use kitty litter or spill pillows to prevent spreading.',
      icon: 'scatter_plot' 
    },
    {
      title: '‎ ‎ ‎ Clean up safely',
      description: 'Use Personal Protective Equipment (PPE) like gloves, goggles, respirators, or boots.',
      icon: 'cleaning_services' 
    },
    {
      title: '‎ ‎ ‎ Preventative Measures',
      description: 'Prioritize routine inspections, Ventilation and tighten connections to Stop leaks',
      icon: 'shield' 
    }
  ];
  
  leaks = [
    {
      title: '‎ ‎ Locate and Stop the source',
      description: 'Use leak detectors and shut off valve to avoid overflow.',
      icon: 'search' 
    },
    {
      title: '‎ ‎ ‎ Assess severity',
      description: 'Estimate the Size, Location and material type to determine the risk',
      icon: 'assessment' 
    },
    {
      title: '‎ ‎ ‎ Call for help',
      description: 'Follow chain of command and Communicate the Leak as soon as possible',
      icon: 'call' 
    },
    {
      title: '‎ ‎ ‎ Preventative Measure',
      description: 'Prioritize routine inspections, Ventilation and tighten connections to Stop leaks',
      icon: 'build' 
    }
  ];
  
  equipment = [
    {
      title: '‎ ‎‎‎ ‎Isolate and Shut down',
      description: 'Press Emergency Stop button to minimize potential damage.',
      icon: 'power_settings_new' 
    },
    {
      title: '‎ ‎ ‎ Secure area',
      description: 'Block entry points and keep people away to prevent harm.',
      icon: 'security' 
    },
    {
      title: '‎ ‎ ‎ Request maintenance',
      description: 'Report issues clearly and swiftly to the expert to minimize downtime.',
      icon: 'build_circle' 
    },
    {
      title: '‎ ‎ ‎ Preventative measures',
      description: 'Enforce proper use protocols and establish preventative maintenance schedules to maximize equipment lifespan.',
      icon: 'event_available' 
    }
  ];
  
  normal = [
    {
      title: '‎ ‎  Personal Protection',
      description: 'Equip workers with respirators, suits, and other gear for protection during mitigation efforts.',
      icon: 'masks' 
    },
    {
      title: '‎ ‎ ‎ Training',
      description: 'Workers should be properly trained on the specific safety procedures for Potential hazards and how to avoid them.',
      icon: 'school' 
    },
    {
      title: '‎ ‎ ‎ Pre-Start Inspections',
      description: 'Before starting any work, perform a thorough inspection of the equipment and workspace to identify any potential hazards.',
      icon: 'engineering' 
    },
    {
      title: '‎ ‎ ‎ Hazard Communication',
      description: 'Be aware of the safety data sheets (SDS) for any chemicals being used and communicate any hazards to others in the area.',
      icon: 'announcement' 
    },
    {
      title: '‎ ‎ ‎ Permits',
      description: ' Obtain necessary permits for working with high-risk equipment or materials.',
      icon: 'assignment' 
    },
    {
      title: '‎ ‎ ‎ Emergency Procedures',
      description: 'Know emergency exits, evacuation routes, and assembly points. Be familiar with how to report fires, accidents, or medical emergencies.',
      icon: 'emergency' 
    }
  ];
  accuracy: any;
  
  
  
  get activeMitigationTechniques() {
    switch(this.incidentType) {
      case 'Spill':
        
        return this.spills;
        
      case 'Leak':
        return this.leaks;
      case 'Equipment Damage':
        return this.equipment;
      case 'Safe Zone (No Incident)':
        return this.normal;
      default:
        return [];
    }
  }
  getMitigationTitle(): string {
    switch(this.incidentType) {
      case 'Spill':
        return 'Spill Mitigation Techniques';
      case 'Leak':
        return 'Leak Mitigation Techniques';
      case 'Equipment Damage':
        return 'Equipment Damage Mitigation Techniques';
      case 'Safe Zone (No Incident)':
        return 'Normal Safety Procedures';
      default:
        return 'Incident Mitigation Techniques';
    }
  }
  
  constructor(private route: ActivatedRoute,private snackBar: MatSnackBar) {}
  
  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      this.videoUrl = params['videoUrl'];
      this.classLabel = params['classLabel'];
      this.incidentType = params['incidentType'];
      this.pred=params['pred'];
      this.test_acu = params['test_acu'];
      this.true_pos = params['true_pos'];
      this.true_neg = params['true_neg'];
      this.false_pos = params['false_pos'];
      this.false_neg = params['false_neg'];
      this.accuracy = params['accuracy'];
      this.precision_good = params['precision_good'];
      this.precision_leaking = params['precision_leaking'];
      this.recall_good = params['recall_good'];
      this.recall_leaking = params['recall_leaking'];
      this.roc = params['roc'];
      console.log('Video URL:', this.videoUrl);
      console.log('Class Label:', this.classLabel);
      console.log('Incident Type:', this.incidentType);
      console.log('Prediction:', this.pred);
      if (this.incidentType === 'Spill') 
      {
        this.snackBar.open('Chemical Spill Reported.\nSupervisor Notified Via Email.', 'Close', {
          duration: 10000,
          verticalPosition: 'top', 
      });
      }
      else if (this.incidentType === 'Leak') 
      {
          this.snackBar.open('Chemical Leak Reported.\nSupervisor Notified Via Email.', 'Close', {
            duration: 10000,
            verticalPosition: 'top', 
        });
      }
      else if (this.incidentType === 'Equipment Damage') 
        {
            this.snackBar.open('Equipment Damage Reported.\nSupervisor Notified Via Email.', 'Close', {
              duration: 10000,
              verticalPosition: 'top', 
          });
        }
    });
  }
}
