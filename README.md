# dsps2024

# https://drive.google.com/drive/folders/1vXhBhl_S94igOTjWkfMkQQXJC9VZ4JIG?usp=sharing

This folder contains all the models we used

- class.pt (YOLOv8s-cls.pt classification model. Take argmax of probabilities to find prediction)
- seg_model.pt (YOLOv8s-seg.pt segmentation model)
- obb_model.pt (YOLO8l-obb.pt oriented object detection model)

- seg_model.keras (Keras model. Expects two images 1) Original image passed through VGG16 2) Segmentation mask made by seg_model.pt passed through VGG16)
- obb_model.keras (Keras model. Expects two images 1) Original image passed through VGG16 2) Segmentation mask made by obb_model.pt on original image passed through VGG16)

In order to run our predictions define the functions in the notebook (not the ones for training) then you can use these functions:

```
model = YOLO('runs/classify/trainX/weights/best.pt')

test_path = Path('test_v2/test/')
rows = []
for tst_img in test_path.glob('**/*.jpg'):
    preds = model(tst_img)
    cls_dict = preds[0].names
    probs = preds[0].probs.data.cpu().numpy()
    pred_pci=int(cls_dict[np.argmax(probs)])
    rows.append({'image_name':os.path.basename(tst_img),
                 'pci':max(0,min(100,pred_pci))})
df_test = pd.DataFrame(rows)

df_test.to_csv("class.csv",header=True)
def gen_submit(df):
  out_json = []
  for idx, results in df.iterrows():
    out_json.append({results['image_name']:results['pci']})
  with open('class.json', 'w') as f:
    json.dump(out_json, f)

df_test['pci'] = df_test['pci'].astype(int)
gen_submit(df_test)

def seg_pred(dir='', name=''):
    path = os.path.join(dir, name)

    features = extract_seg_features(path)
    predictions = seg_model.predict(np.array([features]))
    return (predictions[0])

def obb_pred(dir='', name=''):
    path = os.path.join(dir, name)

    features = extract_obb_features(path, obb_model)
    predictions = obb_model.predict(np.array([features]))
    return (predictions[0])

def gen_submit(df, name='submission.json'):
    out_json = []
    for idx, results in df.iterrows():
        out_json.append({results['image_name']:results['PCI']})
    with open(name, 'w') as f:
        json.dump(out_json, f)

def predict_seg_submission(test_dir):
    names = []
    preds = []
    for img in tqdm(os.listdir(test_dir)):
        pred = seg_pred(test_dir, img)
        
        preds.append(np.argmax(pred))
        names.append(img)
    df = pd.DataFrame({'image_name':names, 'PCI':preds})
    gen_submit(df, 'seg.json')

def predict_obb_submission(test_dir):
    names = []
    preds = []
    for img in tqdm(os.listdir(test_dir)):
        pred = obb_pred(test_dir, img)
        
        preds.append(np.argmax(pred))
        names.append(img)
    df = pd.DataFrame({'image_name':names, 'PCI':preds})
    gen_submit(df, 'obb.json')

predict_seg_submission('test_v2/test')
predict_obb_submission('test_v2/test')
```
