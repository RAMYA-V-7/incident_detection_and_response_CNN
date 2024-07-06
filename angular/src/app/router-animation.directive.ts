import { Directive, HostListener, ElementRef, Renderer2 } from '@angular/core';
import { Router, NavigationStart, NavigationEnd } from '@angular/router';

@Directive({
  selector: '[routerAnimation]'
})
export class RouterAnimationDirective {
  constructor(
    private router: Router,
    private el: ElementRef,
    private renderer: Renderer2
  ) {
    this.router.events.subscribe(event => {
      if (event instanceof NavigationStart) {
        this.renderer.addClass(this.el.nativeElement, 'slide-leave');
        this.renderer.removeClass(this.el.nativeElement, 'slide-enter-active');
      } else if (event instanceof NavigationEnd) {
        this.renderer.addClass(this.el.nativeElement, 'slide-enter');
        this.renderer.removeClass(this.el.nativeElement, 'slide-leave-active');
        setTimeout(() => {
          this.renderer.addClass(this.el.nativeElement, 'slide-enter-active');
          this.renderer.removeClass(this.el.nativeElement, 'slide-enter');
        }, 0);
      }
    });
  }
}
