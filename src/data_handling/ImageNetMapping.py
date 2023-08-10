# Adapted from TorchVision datasets
# The only difference in the code below is the removal of some unneeded transformations
# as well as the handling of paths, 
# which is made to fit the custom path processing of this project

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import scipy.io as sio

def create_imagenet_mapping(data_dir, processed_dir):
    mapping_name = os.path.join(processed_dir,"idx_to_name_mapping.txt")
    if os.path.exists(mapping_name):
        return
    metafile = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    with open(mapping_name, "w") as mapping:
        for idx, wnid in idx_to_wnid.items():
            mapping.write(f"{idx} {wnid}\n")
