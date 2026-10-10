# vendor/ — dependencias empaquetadas

## three-r160.iife.min.js

three.js **r160** (MIT) + `OrbitControls` empaquetados como **script clásico** (IIFE) que expone `window.THREE` (con `THREE.OrbitControls`). Así las animaciones 3D del curso funcionan:
- por `file://`, donde los módulos ES no cargan;
- en GitHub Pages;
- dentro de los iframes del módulo;

y lo hacen sin CDN.

- **Origen:** copia local de `three.module.js` r160 y `examples/jsm/controls/OrbitControls.js`, ya presente en el equipo del docente. Elegida por el docente el 2026-10-10 (auditoría 2026-10).
- **Empaquetador:** rolldown 1.0.3 (el de `dl-explorer/node_modules`, usado sin modificarlo).
- **Entrada** (`entry.js`):
  ```js
  import * as THREE_NS from './three.module.js';
  import { OrbitControls } from './jsm/controls/OrbitControls.js';
  window.THREE = Object.assign({}, THREE_NS, { OrbitControls });
  ```
- **Configuración:** `format: 'iife'`, `minify: true` y `resolve.alias: { three: './three.module.js' }`.
- **Comando:** `rolldown -c rolldown.config.mjs`.

| Archivo | sha256 |
|---|---|
| three.module.js (r160, fuente) | `76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495` |
| jsm/controls/OrbitControls.js (fuente) | `5a44a9e86a2a0fb11933eed69bc2cd33c76a496854c1aed6ed776efa87d7b064` |
| three-r160.iife.min.js (resultado) | `b5b982cacebae08058f7f923e6242dc1cd9d6eeb1d789969ff36187b486e0e50` |

**Licencia:** MIT, en `LICENSE-three.txt`.

**Usado por:** `capa-oculta-3d.html` (3D-2, Módulo 1 · Cap 1.2). Lo usarán también 3D-1, 3D-3 y 3D-4.
