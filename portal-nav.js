/*
 * portal-nav.js — Botón flotante "Volver al portal" para los materiales del curso.
 * Se inserta FUERA del árbol de React (hermano de #root en <body>), así funciona
 * igual en los SPA React y en los HTML estáticos, sin riesgo de romper el render.
 * Incluir con: <script defer src="portal-nav.js"></script>
 * NO incluir en index.html (es el portal) ni en los widgets embebidos por iframe.
 */
(function () {
  'use strict';

  // No mostrar dentro de un iframe embebido (widgets CNN/ViT/CLAHE/AIAS).
  try { if (window.top !== window.self) return; } catch (e) { return; }

  // No mostrar en el portal mismo (defensa por si el tag se incluye allí).
  var path = (location.pathname || '').toLowerCase();
  if (path === '' || path.charAt(path.length - 1) === '/' || /index\.html$/.test(path)) return;

  function mount() {
    if (document.getElementById('portal-nav-btn')) return;

    var css = document.createElement('style');
    css.textContent =
      '#portal-nav-btn{position:fixed;top:14px;right:14px;z-index:9999;' +
      'display:inline-flex;align-items:center;gap:8px;padding:9px 16px;border-radius:9999px;' +
      'background:linear-gradient(135deg,#3D008D,#ED1E79);color:#fff;' +
      'font-family:Montserrat,"Segoe UI",system-ui,sans-serif;font-size:14px;font-weight:700;line-height:1;' +
      'text-decoration:none;box-shadow:0 6px 18px rgba(61,0,141,0.30);' +
      'transition:transform .18s ease,box-shadow .18s ease;-webkit-tap-highlight-color:transparent;}' +
      '#portal-nav-btn:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(61,0,141,0.40);color:#fff;}' +
      '#portal-nav-btn:focus-visible{outline:3px solid #FDB913;outline-offset:2px;}' +
      '#portal-nav-btn svg{width:16px;height:16px;flex:none;display:block;}' +
      '#portal-nav-btn .pn-label{white-space:nowrap;}' +
      '@media print{#portal-nav-btn{display:none;}}' +
      '@media (max-width:480px){#portal-nav-btn{padding:11px;}#portal-nav-btn .pn-label{display:none;}}';
    document.head.appendChild(css);

    var a = document.createElement('a');
    a.id = 'portal-nav-btn';
    a.href = 'index.html';
    a.setAttribute('aria-label', 'Volver al portal del curso');
    a.title = 'Volver al portal del curso';
    a.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v10h14V10"/></svg>' +
      '<span class="pn-label">Portal</span>';
    document.body.appendChild(a);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
