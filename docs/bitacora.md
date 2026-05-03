# Bitácora

Notas de trabajo del proyecto. Voy apuntando aquí decisiones, parámetros
y los pequeños detalles que de aquí a tres meses no me voy a acordar.

---

## 26 abr 2026 — arranque

Punto de partida: tengo el reporte técnico (`ArchPredict_Technical_Report.pdf`)
que define la arquitectura y el roadmap, QGIS 3.44 sobre Fedora, y todas
las dependencias que necesito (numpy, pandas, scikit-learn, rasterio,
joblib) cargan desde la consola Python sin tocar nada. Cero datos
arqueológicos propios.

Decisión metodológica: empezar con datos sintéticos antes de
descargarme rásteres reales. La idea es construir un playground donde
yo conozca la verdad subyacente, así si el RF no recupera la regla que
inventé, sé que el bug está en el pipeline y no en los datos. Cuando el
pipeline esté validado, paso a datos reales.

### DEM sintético

Script `01_create_synthetic_dem.py`. Cuadrícula 500×500 a 30 m, EPSG:25830.
La elevación la compongo así:

- Gradiente lineal NW→SE (amplitud 400 m).
- Río sinuoso tallado con perfil gaussiano alrededor de un eje en
  `sin(2πy·1.5)` (profundidad 80 m).
- Ruido fractal aproximado: suma de campos aleatorios a 4 escalas (8,
  16, 32, 64 px), normalizado a desviación 25 m.

Seed 42 fijado en todo el proyecto. Cargo el TIFF en QGIS, se ve el
gradiente, se ve el río serpenteante y la textura es razonable. Sin
sorpresas.

### Slope y distancia al agua

Script `02_derive_slope_and_distance.py`.

Slope con Horn (1981), 3×3, kernels Sobel-like. Detalle que casi se me
escapa y que apunto aquí para no repetirlo: hay que invertir el signo
del kernel `dz/dy` porque en raster space el índice de fila crece hacia
el sur, pero geográficamente "norte" es arriba. Si no lo haces, la
pendiente sale coherente en magnitud pero la aspect (cuando la calcule
más adelante) iría espejada.

Para la distancia al agua tomo como río los píxeles del percentil 2 más
bajo del DEM y aplico `distance_transform_edt` con `sampling=cellsize`
para que la salida esté ya en metros. No es hidrología "de verdad",
pero como heurística aguanta tanto en el sintético como en DEMs reales
de primera aproximación. Cuando llegue el momento, lo cambio por
acumulación de flujo + umbral.

Visualmente en QGIS: la pendiente sale texturada con escarpes en los
bordes que el ruido fractal genera, la distancia al agua hace ese
patrón de "campo magnético" alrededor del corredor central.

### Sitios sintéticos por regla conocida

Script `03_generate_synthetic_sites.py`. Esta es la parte más
divertida: invento la arqueología.

La regla de asentamiento es multiplicativa, tres preferencias en [0,1]:

```
water_score = exp(-dist_water / 300 m)
slope_score = exp(-slope_deg / 8°)
elev_score  = exp(-(elev - 250)² / (2·80²))

p(site) = water · slope · elev
```

Multiplicativa y no aditiva porque así fallar fuerte en cualquier eje
colapsa la probabilidad. Mucho más realista que una suma lineal donde
puedes "compensar" estar lejos del agua con una pendiente baja.

Sampling: 80 sitios totales, 68 (85 %) sampleados ponderados por p sin
reemplazo, 12 (15 %) puestos uniformemente al azar sobre celdas no
seleccionadas. Esos 12 son ruido humano: factores no observables que el
modelo no podrá predecir (ritual, política, errores de catálogo). Sin
este ruido, el RF saca AUC = 1.0 y me engaño solo.

Por qué 80 sitios: el reporte marca <30 como warning automático y <50
como riesgo de overfit; 80 da margen sin gastar todo el presupuesto.
Más adelante quiero hacer un experimento variando N para caracterizar
cómo degrada el modelo.

Salida en GeoPackage en lugar de Shapefile. Razón: GPKG no trunca
nombres de campo a 10 caracteres, es un solo fichero, y el shapefile es
un formato muerto desde 2014. Los atributos incluyen los scores
parciales y los valores brutos de los predictores para poder auditar.

### Repo

GitHub: `chinijo21/archpredict`, público. Identidad git con el email
noreply para no exponer el personal en commits. `.gitignore` excluye
todo `data/` (con whitelist para `.gitkeep` y `README.md`) y modelos
serializados.

Una vez el repo creado, push del primer commit con `gh auth setup-git`
+ `git push` (la primera vez gh no había configurado el credential
helper de git automáticamente).

### Pseudo-ausencias y dataset de entrenamiento

Script `04_pseudo_absences_and_training_data.py`.

Estrategia B del reporte: estratificación ambiental. Bineo cada
predictor en 4 cuantiles, hago el producto cartesiano (hasta 64
combinaciones), y reparto las pseudo-ausencias proporcionalmente al
número de celdas que cada bin contiene. Dentro de cada bin, sampling
uniforme. Así garantizo que el "fondo" representa el espacio ambiental
del paisaje, no solo las zonas más comunes.

Buffer de exclusión de 150 m (5 píxeles) alrededor de cada presencia
antes de samplear. Sin esto, una ausencia podría caer al lado de una
presencia con valores casi idénticos de los predictores y eso es
leakage de manual.

Ratio absence:presence = 1:1 (80 vs 80). El reporte sugiere 10:1 pero
eso es para MaxEnt/Strategy A; para RF con `class_weight='balanced'` lo
estándar es 1:1 o 1:2.

Salida doble: `synthetic_pseudo_absences.gpkg` para inspección visual
y `training_dataset.csv` directo para sklearn (columnas: point_id, x,
y, presence, elev_m, slope_deg, dist_water_m, is_noise).

Sanity check rápido: `df.groupby('presence').describe()` muestra
medias claramente distintas en `dist_water_m` y `slope_deg` entre las
dos clases, como debería ser.

### Estado al cierre

Cuatro scripts, pipeline sintético completo y reproducible desde
cualquier máquina con seed 42. Todo en el repo. Próximo paso: el
primer Random Forest, con spatial block CV desde el principio (NO
k-fold estándar — esto está marcado en rojo en el roadmap del reporte
y lo apunto aquí también para no olvidarlo nunca).

---

## 27 abr 2026

Empiezo a leer en serio la sección 5 del reporte. Quiero pelearme con
el tema del cross-validation antes de escribir nada, porque ya he visto
demasiados papers donde se reportan AUCs preciosos de 0.95 y luego en
producción no aciertan ni una. La regla del proyecto (no k-fold
estándar) viene de ahí.

El argumento es simple: los sitios están agrupados espacialmente —
porque los humanos no se asentaban uniformemente — entonces si haces un
random split, tu test set acaba con puntos que están a 200 m de puntos
del training. El modelo no está prediciendo en zonas nuevas, está
prácticamente interpolando entre vecinos. El reporte cuantifica el
sesgo: AUC inflado entre 0.10 y 0.25 frente a la performance real
cuando se aplica a un área que el modelo no ha visto.

La alternativa: dividir el área en bloques, cada bloque entero a un
fold, así el test queda geográficamente separado del training. Lo que
no me queda claro todavía es el tamaño de bloque adecuado.

## 28 abr 2026

Sigo dándole vueltas al tamaño de bloque. La literatura "purista"
sugiere derivarlo del rango de autocorrelación del residuo del modelo
(variograma del residuo y leer la distancia donde se estabiliza), pero
eso requiere un primer ajuste del modelo con un CV cualquiera, calcular
residuos, hacer el variograma, y solo entonces redefinir bloques. Es un
proceso iterativo perfectamente válido pero que para una primera
iteración es overkill.

El reporte sugiere "5x la distancia media al vecino más cercano" como
default. Razonable. Pero hay un detalle que me preocupa con N pequeño y
sitios muy clusterizados: la NN media va a ser pequeña porque la regla
los pone cerca del agua. Si pongo bloques de 5×NN-mean voy a tener
bloques pequeños, y eso vacía el sentido de la separación geográfica.

Para el sintético con 80 sitios y AOI 15×15 km, voy a calcular la NN
media a posteriori (informativa) pero usar un suelo absoluto. Si pongo
bloques de 3 km tengo 5×5 = 25 bloques, lo justo para 5-fold. Para
datos reales esto es un parámetro a tunear consciente.

## 30 abr 2026

Implementación de la rejilla. Cada punto cae en su bloque por
`floor((x - xmin) / block_size)` (y lo mismo en y). Bloque
identificado con un id único `bx*10000 + by` por si en datos reales
llegan a 4 dígitos.

Los folds: barajo los IDs de bloque con seed 42 y reparto round-robin a
los 5 folds. Round-robin tras shuffle balancea automáticamente el
número de bloques por fold. Si no es divisible, unos folds tienen un
bloque más que otros, sin más drama.

Cargué los puntos coloreados por fold en QGIS — se ve la separación
espacial, cada fold cubre franjas distintas. Buena pinta.

## 1 may 2026

Bucle de CV implementado. Entreno un RF con los hiperparámetros del
reporte (500 árboles, sqrt features, depth completo, leaf=5, balanced)
sobre 4/5 de los folds y predigo sobre el restante. Para AUC uso el
estándar de sklearn. Para TSS uso `roc_curve` y tomo `max(tpr - fpr)`,
que es algebraicamente equivalente a `max(sensitivity + specificity -
1)` y evita tener que iterar manualmente sobre umbrales.

Una decisión defensiva que añadí mientras programaba: si un fold tiene
0 elementos de cualquier clase en su test set (cosa improbable con
N=80 pero plausible con datos reales sparse), skipeo el fold y reporto
warning en log. El summary solo agrega los folds evaluados. Mejor que
explote AUC por dividir entre cero.

## 2 may 2026

Run completo. El RF aprende la regla razonablemente bien — cosa
esperable porque los predictores son justo los que generan la regla.
Métricas detalladas en `reports/cv_results_synthetic.json`: composición
por fold, AUC y TSS por fold, media y desviación.

Detalle de reproducibilidad que apunto aquí porque me ha llevado un
rato decidirlo: junto con el modelo serializado guardo metadata en el
mismo `.joblib`: versión de sklearn, hiperparámetros, SHA256 (truncado
a 16 caracteres) del CSV de entrenamiento, fecha UTC del entrenamiento
y la semilla. Si en seis meses tengo que demostrar a un reviewer que el
modelo X corresponde al dataset Y, esto lo deja blindado. El reporte lo
exige en la sección 6.5 (reproducibility report).

## 3 may 2026

Cierro el módulo. Modelo final entrenado sobre los 160 puntos completos
en `models/rf_synthetic.joblib`, métricas del CV en su JSON. Las
carpetas `models/` y `reports/` van al gitignore — son artefactos
regenerables desde el script con seed 42. El que clone el repo lo
recompila en un minuto.

Próximo paso (script 06): aplicar el modelo al stack de predictores
para generar el ráster de probabilidad sobre todo el AOI. Y, regla
crítica del proyecto y exigencia explícita del reporte (sección 6.1),
junto al ráster de probabilidad genero el de incertidumbre como
desviación estándar de las predicciones de los árboles individuales
del bosque. Una probabilidad sin barra de error es una probabilidad
que se publica con sobreconfianza, y es justo lo que el reporte critica
de la práctica habitual del campo.
