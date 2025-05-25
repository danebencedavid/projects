# 3D Osztályozás Projekt Dokumentáció

Ez a dokumentáció a 3D osztályozási projekthez készült, amely különböző gépi tanulási módszereket alkalmaz 3D modellek osztályozására a ModelNet10 adathalmazon.

## Adathalmaz

A projekt a [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip) adathalmazt használja, amely 10 különböző kategóriájú 3D modellt tartalmaz. Az adathalmaz automatikusan letöltésre és kibontásra kerül a szkriptek futtatásakor.

```python
DATA_DIR = keras.utils.get_file(
    "modelnet.zip",
    "[http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
print(DATA_DIR)
```

## Feature Készítés SVM-hez és Random Forest-hez
A klasszikus gépi tanulási modellekhez (SVM, Random Forest) topológiai és geometriai jellemzőket nyerünk ki a 3D hálókból. A következő jellemzők kerülnek kinyerésre:
- Bounding box dimenziói ($x, y, z$)
- Felület területe
- Térfogat
- Kompaktság
- Excentricitás
- Genus
- Euler-karakterisztika
- Összefüggő komponensek száma

## Support Vector Machine (SVM)
Különböző kernel függvényekkel (linear, poly, rbf, sigmoid) tanítunk egy SVM modellt a kinyert jellemzőkön.
```python
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    print(f"\nUsing {kernel} kernel")

    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, random_state=42))
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```
## PointNet
A PointNet egy mély tanulási architektúra, amely közvetlenül pontfelhőkön működik.  
Források:
- [Implementation](https://github.com/fxia22/pointnet.pytorch)
- [Official paper](https://arxiv.org/pdf/1612.00593)

A ModelNet10 adathalmazhoz létrehozunk egy `train.txt` és egy `test.txt` fájlt, amely tartalmazza a fájlok elérési útjait. A `.off` fájlokat `.ply` formátumra konvertáljuk, mivel a ModelNetDataset osztály `.ply` fájlokat vár.
```python
with open(os.path.join(d_path, 'train.txt'), 'w') as train_f, \
    open(os.path.join(d_path, 'test.txt'), 'w') as test_f:
    for cls in os.listdir(d_path):
        cls_path = os.path.join(d_path, cls)
        if os.path.isdir(cls_path):
            for split in ['train', 'test']:
                split_path = os.path.join(cls_path, split)
                if os.path.isdir(split_path):
                    for file in os.listdir(split_path):
                        if file.endswith('.ply'):
                            line = f"{cls}/{split}/{file}\n"
                            if split == 'train':
                                train_f.write(line)
                            else:
                                test_f.write(line)

print("Generated train.txt and test.txt successfully.")

os.chdir('pointnet.pytorch')

gen_modelnet_id(d_path)

def convert_off_to_ply(off_path, ply_path):
    mesh = trimesh.load(off_path)
    mesh.export(ply_path)

for root, dirs, files in os.walk(d_path):
    for file in files:
        if file.endswith('.off'):
            off_file = os.path.join(root, file)
            ply_file = off_file.replace('.off', '.ply')
            convert_off_to_ply(off_file, ply_file)

train_dataset = ModelNetDataset(root=d_path, split='train', npoints=1024, data_augmentation=True)
test_dataset = ModelNetDataset(root=d_path, split='test', npoints=1024, data_augmentation=False)
```
Maga az architektúra az alábbi módon műküdik:
- `STN3d` (Spatial Transformer Network)
  Egy 3x3-as transzformációs mátrixot tanul meg a bemeneti pontfelhők (x, y, z) koordinátáinak igazításához. Az STN-t (Spatial Transformer Network) azért használjuk, hogy a PointNet invariáns legyen a geometriai transzformációkkal szemben. 1D konvolúciókat használva a feature-ök kinyeréséhez, teljesen összekötött rétegekkel egy 9 dimenziós vektort jósol meg.
- STN3d-t használ a bemeneti transzformációhoz, és 1D konvolúciókat alkalmaz a lokális feature-ök kinyeréséhez.
- `PointNetfeat` egy modul, amely amely feature-öket von ki pontfelhőkből. Beállítja a szükséges rétegeket: egy `STN3d`-t a bemeneti pontfelhő igazításához, konvolúciós rétegeket `Conv1d`  (3-ról 64-re, majd 64-ről 128-ra, végül 128-ról 1024-re) és batch normalizációs rétegeket `BatchNorm1d` minden konvolúciós réteg után.
- Globális feature-öket kinyer a PointNetfeat-ből, teljesen összekötött rétegeket használ az osztályozáshoz, valamint egy *0.3*-as dropout-ot, a kimenethez pedig softmax-ot alkalmaz.  
![image](https://github.com/user-attachments/assets/f6b03805-3688-46d9-8158-55237492f4ae)

## DGCNN
A `DGCNN` (Dynamic Graph CNN) egy másik pontfelhő alapú mély tanulási architektúra, amely dinamikusan konstruált gráfokat használ a pontok közötti kapcsolatok megragadására.
![image](https://github.com/user-attachments/assets/195ec352-e35d-42b6-ada5-a8b02a24c2cb)


## ZÁRÁS
| Modszer/modell | Pontossag | Bemenet                          |Runtime|
|----------------|-----------|----------------------------------|--------|
| SVM            |     39.07% (rbf) - 40.07% (linear) - 26.46% (poly) - 29.72% (sigmoid)       | Topologiai/Geometriai leirok     |1.2s|
| RandomForest   |     76.63%      | Topologiai/Geometriai leirok     |0.7s|
| PointNet       |     80.08% (train) - 72.47% (test)      | Point Cloud                      | 29m (15 epoch)|
| DGCNN          |     93.08% (train) - 87.33% (test)| Point Cloud, de graf generalodik | 36m (5 epoch)

