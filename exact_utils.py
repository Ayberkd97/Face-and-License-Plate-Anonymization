import cv2
import uuid
import math
import urllib3

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.teams_api import TeamsApi
from exact_sync.v1.api.products_api import ProductsApi

from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.models import ImageSet, Team, Product, AnnotationType, Image, Annotation, AnnotationMediaFile
from exact_sync.v1.rest import ApiException


class ExactHandle:
    def __init__(self, host, user, pw):
        self.config = Configuration()
        self.config.verify_ssl = False
        self.config.host = host
        self.config.username = user
        self.config.password = pw

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.client = ApiClient(self.config)

        self.images_api = ImagesApi(self.client)
        self.image_sets_api = ImageSetsApi(self.client)
        self.annotations_api = AnnotationsApi(self.client)
        self.annotation_types_api = AnnotationTypesApi(self.client)
        self.teams_api = TeamsApi(self.client)
        self.products_api = ProductsApi(self.client)

    def get_annotation_types(self, products):
        # load annotation names, id and store them in dict
        annotation_types = {}
        for product in products:
            for anno_type in self.annotation_types_api.list_annotation_types(product=product).results:
                annotation_types[anno_type.id] = anno_type

        return annotation_types

    def get_product_names(self):
        product_names = {}

        products = self.products_api.list_products().results
        for product in products:
            product_names[product.id] = product.name

        return product_names

    def get_image_set(self, name):
        return self.image_sets_api.list_image_sets(name=name).results

    def get_images(self, image_set, target_folder):
        # download the images in the set if nessesary
        img_sum = sum([len(image_list.images) for image_list in image_set])

        images = []
        with tqdm(total=img_sum, desc="Downloading images") as pbar:
            for image_list in image_set:
                for image_id in image_list.images:
                    # get image informations from server
                    image = self.images_api.retrieve_image(id=image_id)
                    name = image.name
                    image_path = Path(target_folder)/name

                    # if image filed does not exist, download it
                    if image_path.is_file() == False:
                        self.images_api.download_image(id=image_id, target_path=image_path, original_image=True)

                    images.append((image_id, image_path, image.name))
                    pbar.update(1)
        
        return images
    
    def get_annotations(self, image_list, products):
        annotations = []
        product_names = self.get_product_names()

        if type(products[0]) == str:
            new_products = []
            for product in products:
                for num, name in product_names.items():
                    if product == name:
                        new_products.append(num)
            
            products = new_products

        annotation_types = self.get_annotation_types(products)

        anno_list = []
        max_request = 50000
        for image in tqdm(image_list, desc="Loading annotations async"):
            # get the annotations for the current image
            #annos = self.annotations_api.list_annotations(async_req=True, image=image[0], deleted=False, pagination=False, limit=50000).results
            annos = self.annotations_api.list_annotations(async_req=True, image=image[0], deleted=False, pagination=True, limit=max_request)
            anno_list.append((annos, image[0], image[1]))

        for anno_tuple in tqdm(anno_list, desc="Processing annotations"):
            annos = anno_tuple[0].get().results

            if len(annos) == max_request:
                raise Exception("Max request limit is not sufficient")

            # get the relevant informations from the annotation
            for anno in annos:
                if anno.annotation_type in annotation_types:
                    anno_type = annotation_types[anno.annotation_type]
                    product_name = product_names[anno_type.product]
                    annotations.append([anno_tuple[1], anno_tuple[2], anno.annotation_type, anno.vector, anno_type.name, product_name, anno.id, anno.unique_identifier, anno.last_edit_time])

        # create pandas datatframe for easier handling
        return pd.DataFrame(annotations, columns=["Image", "Path", "Type", "Vector", "Label", "Product", "ID", "UUID", "Time"])

    def setup_for_results(self, team_name, image_set_name, product_name, annotation_type_names, anno_type):
        # create team if it does not exist
        teams = self.teams_api.list_teams(name=team_name)
        if teams.count == 0:
            team = Team(name=team_name)
            team = self.teams_api.create_team(body=team) 
        else:
            team = teams.results[0]

        # create imageset if it does not exist
        image_sets = self.image_sets_api.list_image_sets(name=image_set_name)
        if image_sets.count == 0:
            image_set = ImageSet(name=image_set_name, team=team.id)
            image_set = self.image_sets_api.create_image_set(body=image_set)
        else:
            image_set = image_sets.results[0]

        # create product if it does not exist
        products = self.products_api.list_products(name=product_name)
        if products.count == 0:
            product = Product(name=product_name, imagesets=[image_set.id], team=team.id)
            product = self.products_api.create_product(body=product)
        else:
            product = products.results[0]

        # create annotation type
        if anno_type == "poly":
            vector_type = int(AnnotationType.VECTOR_TYPE.POLYGON)
        elif anno_type == "fixed_rect":
            vector_type = int(AnnotationType.VECTOR_TYPE.FIXED_SIZE_BOUNDING_BOX)
        elif anno_type == "rect":
            vector_type = int(AnnotationType.VECTOR_TYPE.BOUNDING_BOX)
        elif anno_type == "line":
            vector_type = int(AnnotationType.VECTOR_TYPE.LINE)
        elif anno_type == "point":
            vector_type = int(AnnotationType.VECTOR_TYPE.POINT)

        # create annotation types if they dont exist
        annotation_types = {}
        for y in annotation_type_names:
            annotation_type_server = self.annotation_types_api.list_annotation_types(name=y, product=product.id)
            if annotation_type_server.count == 0:
                annotation_type = AnnotationType(name=str(y), product=product.id, vector_type=vector_type)
                annotation_type = self.annotation_types_api.create_annotation_type(body=annotation_type)
                annotation_types[y] = annotation_type
            else:
                annotation_types[y] = annotation_type_server.results[0]

        #return {"team": team, "image_set":image_set, "product":product, "annotation_types":annotation_types}

    def get_target(self, team_name, image_set_name):#team_name: exact_handle.get_target("Gehweg Segmentierung", "Gehweg_Selbst")

        team = self.teams_api.list_teams(name=team_name).results[0]
        image_set = self.image_sets_api.list_image_sets(name=image_set_name).results[0]

        annotation_types_raw = self.get_annotation_types(image_set.product_set)
        annotation_types = {}
        for key, value in annotation_types_raw.items():
            annotation_types[value.name] = value

        return {"team": team, "image_set": image_set, "annotation_types": annotation_types}

#target = exact_handle.get_target("Gehweg Segmentierung", "Gehweg_Selbst")
#exact_handle.upload_images(img_list, target)
#img_list = [("Image 1", "C:/Data/my_image.png"), ("Image 2", "C:/Data/your_image,png")]

    def upload_images(self, images, target):#target: where to upload images to
        # load images to server
        img_id_dict = {}
        print("Uploading images")
        for image in tqdm(images):
            ret = self.images_api.list_images(image_set=target["image_set"].id, filename=str(image[1].name))
        
            if ret.count == 0:
                image_type = int(Image.ImageSourceTypes.DEFAULT)
                target_image = self.images_api.create_image(file_path=str(image[1]), image_type=image_type, image_set=target["image_set"].id).results[0]

                img_id_dict[str(image[0])] = target_image.id
            else:
                img_id_dict[str(image[0])] = ret.results[0].id

        return img_id_dict

    def clear_annotations(self, img_id_dict, target):
        print("Collect previous annotations")
        clear_list = []
        with tqdm(total=len(target["annotation_types"])*len(img_id_dict)) as pbar:
            for source_img, target_img in img_id_dict.items():
                for key, value in target["annotation_types"].items():
                    anno_id = value.id

                    anno_list_size = self.annotations_api.list_annotations(annotation_type=anno_id, image=str(target_img)).count
                    anno_list = self.annotations_api.list_annotations(annotation_type=anno_id, image=str(target_img), limit=anno_list_size).results
                    clear_list.extend([str(elem.id) for elem in anno_list])
                    pbar.update(1)

        print("Clear previous annotations")
        for i in tqdm(range(0, len(clear_list), 20)):
            clear_string = ','.join(clear_list[i:min(i+20, len(clear_list))])
            self.annotations_api.multiple_delete(clear_string)

    def upload_annotations(self, annotations, img_id_dict, target):
        # upload new annotations to server
        print("Prepare annotations for upload")
        anno_list = []
        for i in range(len(annotations['Image'])):
            img_id = annotations['Image'][i]
            label = annotations['Label'][i]
            vector = annotations['Vector'][i]
            unique_identifier = str(uuid.uuid4())
    
            annotation_type = target["annotation_types"][str(label)]
            annotation = Annotation(annotation_type=annotation_type.id, vector=vector, image=int(img_id_dict[str(img_id)]), unique_identifier=unique_identifier)
            anno_list.append(annotation)
        
        print("Upload annotations")
        for i in tqdm(range(0, len(anno_list), 100)):
            self.annotations_api.create_annotation(body=anno_list[i:min(i+100, len(anno_list))])

