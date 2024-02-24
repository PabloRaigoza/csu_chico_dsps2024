# dsps2024

# https://drive.google.com/drive/folders/1vXhBhl_S94igOTjWkfMkQQXJC9VZ4JIG?usp=sharing

This folder contains all the models we used

- class.pt (YOLOv8s-cls.pt classification model. Take argmax of probabilities to find prediction)
- seg_model.pt (YOLOv8s-seg.pt segmentation model)
- obb_model.pt (YOLO8l-obb.pt oriented object detection model)

- seg_model.keras (Keras model. Expects two images 1) Original image passed through VGG16 2) Segmentation mask made by seg_model.pt passed through VGG16)
- obb_model.keras (Keras model. Expects two images 1) Original image passed through VGG16 2) Segmentation mask made by obb_model.pt on original image passed through VGG16)

In order to run our predictions

```def seg_pred(dir='', name=''):
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
predict_obb_submission('test_v2/test')```
