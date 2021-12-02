import os

# declaring paths
api_path = 'Tensorflow/models'
model_path ='Api_nevera_AI/nevera/scripts_detec/annotations/ssd_nevera_v3'
pipeline_file = f'{model_path}/pipeline.config'

annotations_path = 'Api_nevera_AI/nevera/scripts_detec/annotations'
record_script = 'Api_nevera_AI/nevera/scripts_detec/generate_tfrecord.py'
IMAGE_PATH = 'Api_nevera_AI/nevera/src/train_images'

# generate tfrecords
!python {record_script} -x {os.path.join(IMAGE_PATH, 'train')} -l {f'{annotations_path}/label_map.pbtxt'} -o {os.path.join(annotations_path, 'train.record')} 
!python {record_script} -x {os.path.join(IMAGE_PATH, 'test')} -l {f'{annotations_path}/label_map.pbtxt'} -o {os.path.join(annotations_path, 'test.record')} 

TRAINING_SCRIPT = os.path.join(api_path, 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, model_path, pipeline_file)
