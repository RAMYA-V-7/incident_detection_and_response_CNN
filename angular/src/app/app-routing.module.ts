import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomepageComponent } from './homepage/homepage.component';
import { PredictionComponent } from './prediction/prediction.component';

const routes: Routes = [
  { path: '', component: HomepageComponent },
  { path: 'prediction', component: PredictionComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
