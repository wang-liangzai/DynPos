import torch
import timm
import model
import torch.nn as nn
from transformers import RobertaModel
from transformers import ViTModel

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

class DynRT(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt):
        super(DynRT, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        if not self.opt["finetune"]:
            freeze_layers(self.bertl_text)
            freeze_layers(self.vit)
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.DynRT(opt)
        self.trar2 = model.TRAR.DynRT2(opt)
        self.roberta_text_linear = torch.nn.Linear(768, 512)
        self.classifier_text = torch.nn.Linear(768, 2)
        self.classifier_image = torch.nn.Linear(768, 2)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"],2)
        )

        # 此处是词性模型的神经网络

        self.dense = nn.Linear(in_features=128, out_features=768)
        self.predicter_layer = nn.Linear(in_features=128, out_features=8)  # 13代表词性的数量
        # self.pos_layer = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=768, out_features=128),
        #     torch.nn.ReLU())
        self.pos_layer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # 增强泛化能力
            nn.LayerNorm(128)  # 稳定训练
        )
        self.relu = torch.nn.ReLU()

        # 门控层
        self.gate_layer = nn.Linear(768 * 2, 768)

        # 注意力层
        self.score_layer = nn.Linear(768, 768)
        self.score_proj = nn.Linear(768, 1)




    def vit_forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:, 1:]

    # forward propagate input  加个神经网络
    def forward(self, input):
        # (bs, max_len, dim) text,text_mask,image,label,pos_tokens; [False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True
        bert_embed_text = self.bertl_text.embeddings(input_ids=input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(self.opt["roberta_layer"]):
            layer_output = self.bertl_text.encoder.layer[i](bert_embed_text)
            # transformers 4 returns a tuple; transformers 5 returns a Tensor.
            bert_text = layer_output if isinstance(layer_output, torch.Tensor) else layer_output[0]
            bert_embed_text = bert_text
        # (bs, grid_num, dim)
        # 这里要对特征进行处理：1 先保存一份文本特征，然后进行降维预测特征; 2 与原来的特征进行相加
        deep_copy = bert_embed_text
        pos_rep = self.pos_layer(deep_copy)  # 降维得到具有代表性的词性特征
        pos_span_logits = self.predicter_layer(pos_rep)  # 得到每一类词性预测的信息，这里返回做loss
        pos_representation = self.dense(pos_rep)  # 再升维，与原来的特征相加
        # bert_embed_text = bert_embed_text + pos_representation

        # # 门控融合词性特征
        # gate = torch.sigmoid(self.gate_layer(torch.cat([bert_embed_text, pos_representation], dim=-1)))
        # bert_embed_text = gate * bert_embed_text + (1 - gate) * pos_representation

        # 假设 hidden_size = 768
        fusion_input = torch.stack([bert_embed_text, pos_representation], dim=2)  # (batch, seq_len, 2, hidden)

        # 权重打分
        scores = torch.tanh(self.score_layer(fusion_input))  # (batch, seq_len, 2, hidden)
        scores = self.score_proj(scores).squeeze(-1)  # (batch, seq_len, 2)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, 2)

        # 加权求和
        bert_embed_text = (fusion_input * weights.unsqueeze(-1)).sum(dim=2)

        # combined_pos = torch.stack((bert_embed_text, pos_representation), dim=-1)
        # bert_embed_text = combined_pos.mean(dim=-1)

        # 下面是处理图像特征
        img_feat = self.vit_forward(input[self.input2])

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text, input[self.input3].unsqueeze(1).unsqueeze(2))
        (out2, lang_emb2, img_emb2) = self.trar2(bert_embed_text, img_feat, input[self.input3].unsqueeze(1).unsqueeze(2))
        combined_out = torch.stack((out1, out2), dim=-1)
        # out1 = combined_out.mean(dim=-1)
        # out1 = out1 + out2
        combined_lang = torch.stack((lang_emb, lang_emb2), dim=-1)
        # lang_emb = combined_lang.mean(dim=-1)
        # lang_emb = lang_emb + lang_emb2
        combined_img = torch.stack((img_emb, img_emb2), dim=-1)
        # img_emb = combined_img.mean(dim=-1)
        # img_emb = img_emb + img_emb2
        out = self.classifier(out1)

        img_emb = self.classifier_image(img_emb)
        lang_emb = self.classifier_text(lang_emb)
        result = self.sigm(out)

        del bert_embed_text, bert_text, img_feat, out1, out
    
        return result, lang_emb, img_emb, pos_span_logits

def build_DynRT(opt, requirements):

    
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"] = "./vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=False)
    # vit = ViTModel.from_pretrained("./vit_base_patch32_224/")
    return DynRT(bertl_text, vit, opt)
