# -*- coding: utf-8 -*-
# ------------------------------------------------------------------ import(s)
import sys
import os
import hashlib
import io
import bottle
import numpy as np
import PIL.Image

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

# ------------------------------------------------------------------- class(s)
# ----------------------------------------------------------------------------
class CModel(object):
    """
    モデルの読み込みと実行用のクラス
    """

    def __init__(self, parameter_file="parameters.h5"):
        """
        Args:
            self (object):
            parameter_file (str):
                NNCで学習させた数値が入ったh5ファイルを指定。
        """

        nn.clear_parameters()
        nn.load_parameters(parameter_file)

        self.x = nn.Variable(shape=(100, 3, 64, 64))
        self.y = network(self.x, None)

    def predict(self, np_image, forward_count=100):
        """
        Args:
            self (object):
            np_image (nparray):

        Returns:
            評価結果
            ve_detectでは 0.0 〜 1.0 の範囲の数値が入った配列を戻します。
        """

        self.x.d = [np_image] * forward_count
        self.y.forward()

        score = 0.0
        for v in self.y.d:
            score += v
        score /= forward_count

        if score > 1.0:
            score = 1.0
        elif score < 0.0:
            score = 0.0

        return score


# ---------------------------------------------------------------- function(s)
# ============================================================================
# 以下はNNCから出力したネットワーク構成のスクリプトとなります。
# （評価に使用するため、SquaredErrorの箇所はコメントアウトしてあります。
def network(x, y_index, test=False):
    # Input -> 3,64,64
    # Convolution -> 16,31,31
    with nn.parameter_scope('Convolution'):
        h = PF.convolution(x, 16, (3,3), (0,0), (2,2))
    # Tanh
    h = F.tanh(h)
    # MaxPooling -> 16,16,11
    h = F.max_pooling(h, (2,3), (2,3))
    # Dropout
    if not test:
        h = F.dropout(h)
    # Convolution_2 -> 32,6,5
    with nn.parameter_scope('Convolution_2'):
        h = PF.convolution(h, 32, (5,3), (0,0), (2,2))
    # ReLU_4
    h = F.relu(h, True)
    # MaxPooling_2 -> 32,3,3
    h = F.max_pooling(h, (2,2), (2,2))
    # Dropout_2
    if not test:
        h = F.dropout(h)
    # Convolution_3 -> 64,1,1
    with nn.parameter_scope('Convolution_3'):
        h = PF.convolution(h, 64, (3,3), (0,0), (2,2))
    # Tanh_2
    h = F.tanh(h)
    # Dropout_3
    if not test:
        h = F.dropout(h)
    # Affine -> 50
    with nn.parameter_scope('Affine'):
        h = PF.affine(h, (50,))
    # ReLU_2
    h = F.relu(h, True)
    # Dropout_4
    if not test:
        h = F.dropout(h)
    # Affine_2 -> 5
    with nn.parameter_scope('Affine_2'):
        h = PF.affine(h, (5,))
    # ELU
    h = F.elu(h)
    # Affine_3 -> 1
    with nn.parameter_scope('Affine_3'):
        h = PF.affine(h, (1,))
    # SquaredError
    #h = F.squared_error(h, y_index)
    return h


# ============================================================================
@bottle.route("/")
@bottle.route("/index")
@bottle.route("/index.html")
def html_index():
    return bottle.template("templates/index")

@bottle.route("/res_image/<img_filepath:path>", name="res_image")
def res_image(img_filepath):
    return bottle.static_file(img_filepath, root="./res_image")

@bottle.route("/ein_image/<img_filepath:path>", name="ein_image")
def ein_image(img_filepath):
    return bottle.static_file(img_filepath, root="./ein_image")

@bottle.route("/decide")
def html_decide():
    return bottle.template("templates/index")

@bottle.route("/decide", method="POST")
def do_upload():
    try:
        upload = bottle.request.files.get('upload', '')
        if not upload.filename.lower().endswith((".png", "jpg", ".jpeg")):
            return "File extension not allowed!"
    except AttributeError:
        return bottle.template("templates/index")

    data_raw = upload.file.read()

    o = CModel()

    s = PIL.Image.open(io.BytesIO(data_raw))
    s.thumbnail((64, 64))

    i = PIL.Image.new("RGB", (64, 64))
    x = (64 - i.size[0]) >> 1
    y = (64 - i.size[1]) >> 1

    i.paste(s, (x, y))
    image = np.array(i.getdata(), dtype=np.float32) / 1.0
    image = image.reshape((64, 64, 3))
    image = image.transpose(2, 0, 1)

    score = o.predict(image)
    face = int((score * 100) / 16.7)
    if face > 5:
        face = 5
    elif face < 0:
        face = 0

    filename = "%02d_%s.png" % (face, hashlib.sha1(data_raw).hexdigest())
    if os.path.exists("res_image/" + filename) is False:
        s = PIL.Image.open(io.BytesIO(data_raw))
        s.save("res_image/" + filename)

    return bottle.template(
        "templates/decide",
        filename=filename,
        ein_face="%02d.png" % (face,),
        score="%.3f" % (score * 100,)
    )


if __name__ == "__main__":
    bottle.run(host="localhost", port=8000, debug=True, reloader=True)

# [EOF]
