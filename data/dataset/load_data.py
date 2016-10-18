import NUSwide


from helper.config import config

def load_nus_db(image_set_path):
	nus = NUSwide(image_set_path)
	nusdb = nus.get_db()
	return nus,nusdb