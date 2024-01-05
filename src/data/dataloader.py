import os
import glob
import logging
import numpy as np
import albumentations

from datasets import load_dataset, Dataset

from .postprocessing import convert_obj_to_coco_format


logger = logging.info(__name__)



class DataProcessor:
    def __init__(self, image_processor, data_args) -> None:
        self.image_processor = image_processor
        self.data_args = data_args
        
        with open(self.data_args.object_path, 'r') as file:
            self.id2label = file.readlines()   
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        self.transform = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )
        
    def __call__(self):
        datasets = {}
        # train set
        if self.data_args.train_path_list is not None:
            train_data = self.load_dataset(self.data_args.train_path_list, 'train')
            
            if self.data_args.max_train_samples is not None:
                train_data = train_data.select(range(self.data_args.max_train_samples))
            
            datasets['train'] = self.process_fn(train_data)
            
        # validation set
        if self.data_args.validation_path_list is not None:
            valid_data = self.load_dataset(self.data_args.validation_path_list, 'train')
        
            if self.data_args.max_eval_samples is not None:
                valid_data = valid_data.select(range(self.data_args.max_eval_samples))
                
            datasets['validation'] = self.process_fn(valid_data)
        
        return datasets
    
    def load_dataset(self, data_path:str=None, key:str='train') -> Dataset:
        """ Load datasets function 

        Args:
            data_path (str, optional): folder contain list of input files name. Defaults to None.
            key (str, optional): help dataloader know is train file or test file. 
                                Input file can be train/validation/test. Defaults to 'train'.

        Raises:
            Exception: _description_

        Returns:
            Datasets
        """
        if not os.path.exists(data_path):
            raise ValueError(f'Not found {data_path} path.')
        
        files = glob.glob(os.path.join(data_path, '*'))
        extention = files[0].split('.')[-1]
    
        try:
            data_file = f"{data_path}/*.{extention}"
            
            if self.data_args.streaming:
                datasets = load_dataset(
                    extention, data_files=data_file, split=key, streaming=self.data_args.streaming
                )
            else:
                datasets = load_dataset(
                    extention, data_files=data_file, split=key, 
                    num_proc=self.data_args.num_workers
                )   
                
            # convert format to coco
            dataset = dataset.map(lambda example:convert_obj_to_coco_format(example),
                                  remove_columns=['bboxes', 'labels', 'seg'],
                                  num_proc=self.data_args.num_workers)
            return datasets
        except:
            logger.info(f'Error loading dataset {data_path} with {extention} extention')
    
    # transforming a batch
    def process_fn(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]
        
        result = self.image_processor(images=images, annotations=targets, return_tensors="pt")
        
        return result
    
    def formatted_anns(self, image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations


class DataCollator:
    image_processor = None
    
    def __call__(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]

        features = {}

        features["pixel_values"] = encoding["pixel_values"]
        features["pixel_mask"] = encoding["pixel_mask"]
        features["labels"] = labels

        return features