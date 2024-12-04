from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from model.classifier import Classifier
from dataset.mnist import MNISTDataModule
import torch
from model.vs_module import VideoSumModule
from dataset.vsdataset import VideoSumDataModule
from model.layers.pgl_sum import PGL_SUM
torch.set_float32_matmul_precision('high')

def cli_main():
    cli = LightningCLI(Classifier, MNISTDataModule)


def test():
    from model.layers.pgl_sum import PGL_SUM
    from dataset.vsdataset import VideoData

    model = PGL_SUM(
        input_size=1024,
        output_size=1024,
        freq=10000,
        pos_enc="absolute",
        num_segments=4,
        heads=8,
        fusion="add",
    )
    dataset = VideoData(
        mode="train", root_path="data_source", dataset_name="SumMe", split_index=0
    )

    frame_features, gtscore = dataset[0]
    out, attn = model(frame_features)
    print(out)
    print(attn.shape)
    print(gtscore.shape)


def train():
    root_path = "data_source"

    datamodule = VideoSumDataModule(
        root_path=root_path,
        dataset_name="SumMe",
        split_index=1,
        batch_size=1,
        num_workers=7,
    )

    model = PGL_SUM(
        input_size=1024,
        output_size=1024,
        freq=10000,
        pos_enc="absolute",
        num_segments=4,
        heads=8,
        fusion="add",
    )

    module = VideoSumModule(model=model, T_max=10, eta_min=0)
    trainer = Trainer(max_epochs=10)
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    train()
