import os.path as osp
import shutil
import json
import tensorflow as tf
import functools
from alive_progress import alive_bar
import meshio

def parse(proto, meta: dict) -> dict:
        """Parses a trajectory from tf.Example."""
        feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta['features'].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
            data = tf.reshape(data, field['shape'])
            out[key] = data
        return out

if __name__ == '__main__':
    raw_dir = "raw"
    xdmf_dir = "xdmf"
    sets = ["train", "valid", "test"]
    num_time_steps = 600

    with open(osp.join(raw_dir, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())

    traj=0
    with alive_bar(total=1200) as bar:
        for set in sets:
            # convert data to dict
            ds = tf.data.TFRecordDataset(osp.join(raw_dir, f"{set}.tfrecord"))
            ds = ds.map(functools.partial(parse, meta=meta), num_parallel_calls=1)
            ds = ds.prefetch(1)
            frame=0
            for data in ds:
                with meshio.xdmf.TimeSeriesWriter(osp.join(xdmf_dir, f"cylinder_{traj}.xdmf")) as writer:
                    points = data["mesh_pos"].numpy()[0,:,:]
                    cells = {"triangle": data["cells"].numpy()[0,:,:]}
                    writer.write_points_cells(points, cells)

                    node_type = data["node_type"].numpy()[0,:,0]
                    for t in range(num_time_steps):
                        point_data = {}
                        point_data["velocity"] = data["velocity"].numpy()[t]
                        point_data["pressure"] = data["pressure"].numpy()[t,:,0]
                        point_data["node_type"] = node_type
                        writer.write_data(t, point_data)

                shutil.move(src=f"cylinder_{traj}.h5", dst=osp.join(xdmf_dir, f"cylinder_{traj}.h5"))
                traj+=1
                bar()