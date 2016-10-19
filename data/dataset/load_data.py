
from helper.config import config
from data.dataset.NUSwide import NUSwide


def load_nus_db(image_set_path,root_path):
	nus = NUSwide(image_set_path,root_path)
	nusdb = nus.get_db()
	return nus,nusdb