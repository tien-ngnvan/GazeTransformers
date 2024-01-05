import ast

def convert_str_to_list(x):
    return ast.literal_eval(x)

# convert COCO format
def convert_obj_to_coco_format(example):
  ls_label = convert_str_to_list(example['labels'])
  ls_bbox = convert_str_to_list(example['bboxes'])

  area, bbox, category = [], [], []

  for _, (label, box) in enumerate(zip(ls_label, ls_bbox)):
    xmin, ymin, xmax, ymax = box

    o_width = xmax - xmin
    o_height = ymax - ymin

    area.append(o_width * o_height)
    bbox.append([xmin, ymin, o_width, o_height])
    category.append(label)

  return {
      'image_id' : example['seg'].split('.')[0],
      'objects' : {
          'area' : area,
          'bbox' : bbox,
          'category' : category
      }
  }