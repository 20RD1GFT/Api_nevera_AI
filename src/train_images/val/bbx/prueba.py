# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r'C:\Users\JIOY\OneDrive - GFT Technologies SE\Desktop\Pruebas y notas\Pruebas de arquitecturas\Api_nevera_AI\src\train_images\val\bbx\picture20211112143811.json'

# set path for coco json to be saved
save_json_path = r'C:\Users\JIOY\OneDrive - GFT Technologies SE\Desktop\Pruebas y notas\Pruebas de arquitecturas\Api_nevera_AI\src\train_images\val\picture20211112143811.json'

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)