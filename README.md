# Cold Drinks - Detecao de Objetos

Repositorio com notebook `Cold_drinks.ipynb` treinando detectores para 6 refrigerantes (Coca Cola, Sprite, Pepsi, Mountain Dew, 7UP, Fanta) usando o dataset [Kaggle cold-drinks-inventory-dataset](https://www.kaggle.com/datasets/faseeh001/cold-drinks-inventory-dataset). O fluxo foi pensado para Google Colab com GPU.

## Ambiente e dependencias
- Python 3.10+ com GPU (Colab recomendado).
- Pacotes principais: `kaggle`, `ultralytics`, `torch`, `torchvision`, `pycocotools`, `matplotlib`, `pandas`, `tqdm`.
- Instale no inicio do notebook:
  ```bash
  pip install kaggle ultralytics torch torchvision pycocotools matplotlib pandas tqdm -q
  ```

## Passo a passo resumido
1) Upload do `kaggle.json` e configuracao da API (`~/.kaggle/kaggle.json`).  
2) Download e extracao do dataset (`kaggle datasets download -d faseeh001/cold-drinks-inventory-dataset`) em `cold_drinks/Finalize`.  
3) Geracao do `data.yaml` apontando para `/content/cold_drinks/Finalize` e definindo as 6 classes.  
4) **YOLO11n**  
   - Hiperparametros em `args.yaml` (50 epocas, batch 16, imgsz 640).  
   - Treino com `YOLO('yolo11n.pt').train(data='data.yaml', cfg='args.yaml', name='yolov11')`.  
   - Metricas de validacao: precisao 0.764, recall 0.919, mAP50 0.884, mAP50-95 0.496.  
   - Predicoes salvas em `cold_drinks/Finalize/predict_yolov11/` e teste unico em `test_pred_v11/pred/`.  
5) **YOLOv8n**  
   - Hiperparametros em `args_yolo8.yaml` (mesmos valores de YOLO11).  
   - Treino com `YOLO('yolov8n.pt').train(..., name='yolov8')`.  
   - Metricas de validacao: precisao 0.843, recall 0.796, mAP50 0.922, mAP50-95 0.526.  
   - Predicoes salvas em `cold_drinks/Finalize/predict_yolov8/` e teste unico em `test_pred_v8/pred/`.  
6) **Faster R-CNN (ResNet50 FPN)**  
   - Conversao de rotulos YOLO para formato COCO (`coco_val.json`) para avaliacao com `COCOeval`.  
   - Treino por 20 epocas; pesos em `faster_rcnn_cold_drinks.pth`.  
   - Metricas por classe (mAP50): Coca Cola 0.189, Sprite 0.124, Pepsi 0.343, Mountain Dew 0.380; 7UP/Fanta sem deteccoes.  
   - Predicoes: imagens em `cold_drinks/Finalize/predict_faster_rcnn/`, JSON em `test_pred_json_faster_rcnn/`.  
7) Grafico comparativo de mAP50-95 ao longo do treino para YOLOv8, YOLO11 e valores de referencia do Faster R-CNN.

## Resultados rapidos
- YOLOv8n: mAP50-95 0.526 (melhor resultado geral).  
- YOLO11n: mAP50-95 0.496.  
- Faster R-CNN: mAP50 modesto e classes 7UP/Fanta sem deteccao (dataset pequeno).

## Estrutura gerada (principal)
- `runs/train/yolov11/` e `runs/train/yolov8/`: treinamentos e pesos `best.pt/last.pt`.  
- `cold_drinks/Finalize/predict_yolov11/` e `cold_drinks/Finalize/predict_yolov8/`: predicoes em lote.  
- `test_pred_v11/pred/`, `test_pred_v8/pred/`, `test_pred_faster_rcnn/pred/`: exemplos individuais.  
- `test_pred_json_v11/`, `test_pred_json_faster_rcnn/`, `predict_yolov8/detections.json`: saidas JSON.  
- `faster_rcnn_cold_drinks.pth`, `metrics_fasterrcnn.csv`, `metrics_fasterrcnn.json`: artefatos do Faster R-CNN.

## Como reproduzir
Execute o notebook `Cold_drinks.ipynb` de cima para baixo em um ambiente com GPU. Ajuste caminhos se nao estiver no Colab e, se desejar, edite os YAMLs de hiperparametros para novos testes.
