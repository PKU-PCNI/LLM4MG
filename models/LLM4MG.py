import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.ops import Voxelization

from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from models.ViT import FeatureExtractorViT
from models.radar_encoder import RadarBEVNet
from models.ECAResNet import Bottleneck, ECA_ResNet

from transformers import LlamaModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class PointNet2_ssg(nn.Module):
    def __init__(self, target_HW, normal_channel=False):
        super(PointNet2_ssg, self).__init__()
        self.target_HW = target_HW
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=2, nsample=8, in_channel=6 + additional_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=target_HW * target_HW, radius=4, nsample=16, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        batch_size, dim, num_points = l2_points.shape
        feature_map = l2_points.reshape(batch_size, dim, self.target_HW, self.target_HW)
        reshaped_feature_map = feature_map.reshape(feature_map.shape[0], 64, 4, feature_map.shape[2],
                                                   feature_map.shape[3])
        output = torch.max(reshaped_feature_map, dim=2).values
        return output

class res_classifier(nn.Module):
    def __init__(self, input_f, output, hidden=512, dropout_p=0.3):
        super(res_classifier, self).__init__()
        self.fc1 = nn.Linear(input_f, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.classifier = nn.Linear(hidden, output)
        if input_f != hidden:
            self.shortcut = nn.Linear(input_f, hidden)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout1(out)
        out = self.fc2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        out = self.dropout2(out)

        return self.classifier(out)

class res_reghead(nn.Module):
    def __init__(self, input_f, output, hidden=512, dropout_p=0.3):
        super(res_reghead, self).__init__()
        self.fc1 = nn.Linear(input_f, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.last = nn.Linear(hidden, output)

        if input_f != hidden:
            self.shortcut = nn.Linear(input_f, hidden)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout1(out)
        out = self.fc2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        out = self.dropout2(out)
        return self.last(out)

class LLM4MG(nn.Module):
    def __init__(self, normal_channel=False,
                 point_cloud_range=None,
                 radar_voxel_size=None,
                 vit_inputsize=(320, 480),
                 vit_inchannel=4,
                 vit_patchsize=(20, 30),
                 llama_path=".",
                 input_text='Frequency=60Ghz, Bandwidth=20Mhz, Distance=',
                 LoRA = True,
                 decoder_num = 8,
                 ):
        super(LLM4MG, self).__init__()
        assert vit_inputsize[0] / vit_patchsize[0] == vit_inputsize[1] / vit_patchsize[1], "ViT patch size error."

        self.pointnet_bs = PointNet2_ssg(normal_channel=normal_channel, target_HW=vit_inputsize[0] // vit_patchsize[0])
        self.pointnet_vh = PointNet2_ssg(normal_channel=normal_channel, target_HW=vit_inputsize[0] // vit_patchsize[0])

        self.vit_bs = FeatureExtractorViT(
            image_size=vit_inputsize,
            channels=vit_inchannel,
            patch_size=vit_patchsize,
            dim=191,
            depth=2,
            heads=8,
            mlp_dim=1024,
            dropout=0.2
        )
        self.vit_vh = FeatureExtractorViT(
            image_size=vit_inputsize,
            channels=vit_inchannel,
            patch_size=vit_patchsize,
            dim=191,
            depth=2,
            heads=8,
            mlp_dim=1024,
            dropout=0.2
        )

        self.radar_voxel_size = [0.5, 0.5, 0.3]
        self.point_cloud_range = [0, -15, -0.3, 30, 15, 0.3]
        self.radar_voxel_layer = Voxelization(max_num_points=8,
                                              voxel_size=self.radar_voxel_size,
                                              max_voxels=(900, 1200),
                                              point_cloud_range=self.point_cloud_range)

        self.radarnet_bs = RadarBEVNet(return_rcs=False,
                                       in_channels=3,
                                       feat_channels=[32, 64, 256],
                                       with_distance=False,
                                       point_cloud_range=self.point_cloud_range,
                                       voxel_size=self.radar_voxel_size,
                                       norm_cfg=dict(
                                           type='BN1d',
                                           eps=1.0e-3,
                                           momentum=0.01),
                                       with_pos_embed=True
                                       )
        self.radarnet_vh = RadarBEVNet(return_rcs=False,
                                    in_channels=3,
                                    feat_channels=[32, 64, 256],
                                    with_distance=False,
                                    point_cloud_range=self.point_cloud_range,
                                    voxel_size=self.radar_voxel_size,
                                    norm_cfg=dict(
                                        type='BN1d',
                                        eps=1.0e-3,
                                        momentum=0.01),
                                    with_pos_embed=True
                                    )

        self.ECAResNet_bs = ECA_ResNet(Bottleneck, [1, 2, 2, 1], in_channels=256)
        self.ECAResNet_vh = ECA_ResNet(Bottleneck, [1, 2, 2, 1], in_channels=256)

        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_text = input_text

        self.llama = LlamaModel.from_pretrained(llama_path)
        self.llama_embed = self.llama.embed_tokens
        del self.llama.embed_tokens

        self.llama.layers = self.llama.layers[:decoder_num]

        if LoRA :
            target_modeles = ['down_proj', 'gate_proj']
            peft_config = LoraConfig(
                r=8,  
                lora_alpha=32,
                target_modules=target_modeles, 
                lora_dropout=0.4,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.llama = get_peft_model(self.llama, peft_config)

        self.classifier = res_classifier(input_f=2048, hidden=512, output=2)
        self.power_head = res_reghead(input_f=2048, hidden=512, output=6)
        self.tau_head = res_reghead(input_f=2048, hidden=512, output=6)

    def get_radar(self,x_radar):
        x_split = torch.unbind(x_radar, dim=0)

        voxels_list = []
        coordinates_list = []
        num_points_list = []
        num_voxels_list = []

        for xi in x_split:
            non_zero_mask = torch.any(xi != 0, dim=-1)
            xi_trimmed = xi[non_zero_mask]
            if xi_trimmed.numel() == 0:
                xi_trimmed = torch.zeros((1, 3), device=xi.device)
            voxels, coordinates, num_points = self.radar_voxel_layer(xi_trimmed)

            num_voxels_list.append(voxels.shape[0])
            voxels_list.append(voxels)
            coordinates_list.append(coordinates)
            num_points_list.append(num_points)

        padded_voxels_list = []
        padded_coordinates_list = []
        padded_num_points_list = []
        target_num_voxels = 256
        for i in range(len(voxels_list)):
            voxels = voxels_list[i]
            coordinates = coordinates_list[i]
            num_points = num_points_list[i]

            current_voxels = voxels.shape[0]
            pad_size = target_num_voxels - current_voxels

            if (pad_size <= 0):
                padded_voxels_list.append(voxels[:target_num_voxels])
                padded_coordinates_list.append(coordinates[:target_num_voxels])
                padded_num_points_list.append(num_points[:target_num_voxels])
                continue

            padded_voxels = F.pad(voxels, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
            padded_voxels_list.append(padded_voxels)

            padded_coordinates = F.pad(coordinates, (0, 0, 0, pad_size), mode='constant', value=0)
            padded_coordinates_list.append(padded_coordinates)

            padded_num_points = F.pad(num_points, (0, pad_size), mode='constant', value=0)
            padded_num_points_list.append(padded_num_points)

        voxels = torch.stack(padded_voxels_list, dim=0)
        coordinates = torch.stack(padded_coordinates_list, dim=0).to(x_radar.device)

        num_voxels = torch.tensor(num_voxels_list, dtype=torch.long).to(x_radar.device)

        features = voxels.mean(dim=2).to(x_radar.device)
        return features, num_voxels, coordinates

    def forward(self, x):

        x_rgb_bs = x['view0_RGB']
        x_lidar_bs = x['view0_Lidar']
        x_radar_bs = x['view0_Radar']
        x_rgb_vh = x['view1_RGB']
        x_lidar_vh = x['view1_Lidar']
        x_radar_vh = x['view1_Radar']

        x_lidar_bs = self.pointnet_bs(x_lidar_bs)
        x_lidar_vh = self.pointnet_vh(x_lidar_vh)

        x_rgb_bs = self.vit_bs(x_rgb_bs)
        x_rgb_vh = self.vit_vh(x_rgb_vh)

        bs_f, bs_nv, bs_coor = self.get_radar(x_radar_bs)
        vh_f, vh_nv, vh_coor = self.get_radar(x_radar_vh)
        bs_nv[bs_nv==0] += 1
        vh_nv[vh_nv == 0] += 1

        x_radar_bs = self.radarnet_bs(bs_f, bs_nv, bs_coor)
        x_radar_bs, _ = torch.max(x_radar_bs, dim=1, keepdim=True)
        x_radar_vh = self.radarnet_vh(vh_f, vh_nv, vh_coor)
        x_radar_vh, _ = torch.max(x_radar_vh, dim=1, keepdim=True)

        combined_bs = torch.cat([x_lidar_bs, x_rgb_bs, x_radar_bs], dim=1)
        combined_vh = torch.cat([x_lidar_vh, x_rgb_vh, x_radar_vh], dim=1)

        x_bs = self.ECAResNet_bs(combined_bs)
        B = x_bs.shape[0]
        x_vh = self.ECAResNet_vh(combined_vh)
        combined = torch.cat([x_bs, x_vh], dim=1)
        combined = combined.permute(0, 2, 3, 1).reshape(B, 49, 2048)

        text_input = []
        for i in range(B):
            text = self.input_text + str(x['dis'][i].cpu().item()) + 'm, phi=' + str(x['phi'][i].cpu().item()) +',theta=' + str(x['theta'][i].cpu().item())
            text_input.append(text)

        text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True).to(combined.device)
        text_encode = self.llama_embed(text_input['input_ids']).to(combined.device)
        combined = torch.cat([combined, text_encode], dim=1).to(combined.device)

        llama_outputs = self.llama(inputs_embeds=combined)
        features = llama_outputs.last_hidden_state

        features = torch.mean(features, dim=1, keepdim=True).permute(0, 2, 1).squeeze()

        Results = {}
        Results['power'] = self.power_head(features)
        Results['tau'] = self.tau_head(features)
        Results['LoS'] = self.classifier(features)

        if B == 1:
            Results['LoS'] = Results['LoS'].unsqueeze(0)

        return Results



if __name__ == '__main__':
    model = LLM4MG(llama_path="./Llama-3.2-1B")

