import torch
from persite_painn.nn.modules import (
    EmbeddingBlock,
    FullyConnected,
    MessageBlock,
    ReadoutBlock,
    UpdateBlock,
)
from persite_painn.utils.tools import get_rij, make_directed
from torch import nn


class Painn(nn.Module):
    def __init__(self, modelparams, **kwargs):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        self.modelparams = modelparams
        super().__init__()

        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        readout_dropout = modelparams.get("readout_dropout", 0)
        fc_dropout = modelparams.get("fc_dropout", 0)
        self.means = modelparams.get("means")
        self.stddevs = modelparams.get("stddevs")
        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    feat_dim=feat_dim,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    learnable_k=learnable_k,
                    dropout=conv_dropout,
                )
                for _ in range(num_conv)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(
                    feat_dim=feat_dim, activation=activation, dropout=conv_dropout
                )
                for _ in range(num_conv)
            ]
        )
        # Fully connected layers
        self.multifidelity = kwargs["multifidelity"]
        output_atom_fea_dim = modelparams["atom_fea_len"]
        n_h = modelparams["n_h"]
        h_fea_len = modelparams["h_fea_len"]
        n_outputs = modelparams["n_outputs"]
        activation_f = modelparams["activation_f"]

        if self.multifidelity:
            self.readout_block = ReadoutBlock(
                feat_dim=feat_dim,
                output_atom_fea=output_atom_fea_dim["atom_emb"],
                output_keys=["atom_emb"],
                activation=activation,
                dropout=readout_dropout["atom_emb"],
            )
            self.readout_block_target = ReadoutBlock(
                feat_dim=output_atom_fea_dim["atom_emb"],
                output_atom_fea=output_atom_fea_dim["target"],
                output_keys=["target"],
                activation=activation,
                dropout=readout_dropout["target"],
            )

            self.fn_fidelity = FullyConnected(
                output_atom_fea_dim=output_atom_fea_dim["atom_emb"],
                h_fea_len=h_fea_len["fidelity"],
                n_h=n_h["fidelity"],
                activation=activation_f,
                n_outputs=n_outputs["fidelity"],
                dropout=fc_dropout["fidelity"],
                output_key="fidelity",
                scale=True,
                means=self.means,
                stddevs=self.stddevs,
            )
            self.fn_target = FullyConnected(
                output_atom_fea_dim=output_atom_fea_dim["target"]
                + n_outputs["fidelity"],
                h_fea_len=h_fea_len["target"],
                n_h=n_h["target"],
                activation=activation_f,
                n_outputs=n_outputs["target"],
                dropout=fc_dropout["target"],
                output_key="target",
                scale=True,
                means=self.means,
                stddevs=self.stddevs,
            )
        else:
            self.readout_block = ReadoutBlock(
                feat_dim=feat_dim,
                output_atom_fea=output_atom_fea_dim["target"],
                output_keys=["target"],
                activation=activation,
                dropout=readout_dropout["target"],
            )
            self.fn = FullyConnected(
                output_atom_fea_dim=output_atom_fea_dim["target"],
                h_fea_len=h_fea_len["target"],
                n_h=n_h["target"],
                activation=activation_f,
                n_outputs=n_outputs["target"],
                dropout=fc_dropout["target"],
                output_key="target",
                scale=True,
                means=self.means,
                stddevs=self.stddevs,
            )

        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):
        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

        return s_i, xyz, r_ij, nbrs

    def run(self, batch, xyz=None, inference=False):
        s_i, _, _, _ = self.atomwise(batch=batch, xyz=xyz)
        atomwise_out = self.readout_block(s_i=s_i)
        results = {}
        for key, val in atomwise_out.items():
            out = self.fn(val)
            results[key] = out

        if inference:
            # import here to avoid circular imports
            from persite_painn.utils.cuda import batch_detach

            results = batch_detach(results)

        return results

    def forward(self, batch, xyz=None, inference=False, **kwargs):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results = self.run(batch=batch, xyz=xyz, inference=inference)

        return results

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PainnMultifidelity(Painn):
    def __init__(self, modelparams, **kwargs):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        super().__init__(modelparams, **kwargs)

    def run(self, batch, xyz=None, inference=False):

        s_i, xyz, _, _ = self.atomwise(batch=batch, xyz=xyz)
        atomwise_out = self.readout_block(s_i=s_i)

        # atomwise_out_fidelity = self.readout_block_fidelity(s_i=atom_emb)
        atomwise_out_target = self.readout_block_target(s_i=atomwise_out["atom_emb"])
        results = {}
        
        # fidelity
        fidelity = self.fn_fidelity(atomwise_out["atom_emb"])
        results["fidelity"] = fidelity

        std = self.stddevs["fidelity"].to(s_i)
        mean = self.means["fidelity"].to(s_i)

        for key, val in atomwise_out_target.items():
            fidelity_normed = (fidelity - mean) / std
            new_val = torch.cat((fidelity_normed, val), dim=1)
            out = self.fn_target(new_val)
            results[key] = out

        if inference:
            # import here to avoid circular imports
            from persite_painn.utils.cuda import batch_detach

            results = batch_detach(results)

        return results

    # def atomwise_pretrained_features(self, batch, xyz=None):
    #     nbrs, _ = make_directed(batch["nbr_list"])
    #     nxyz = batch["nxyz"]

    #     if xyz is None:
    #         xyz = nxyz[:, 1:]
    #         xyz.requires_grad = True

    #     z_numbers = nxyz[:, 0].long()
    #     num_atoms = z_numbers.shape[0]

    #     # get r_ij including offsets and excluding
    #     # anything in the neighbor skin
    #     self.set_cutoff()
    #     r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

    #     s_i = self.embed_targ_act(self.embed_block_targ(batch["new_features"]))
    #     v_i = torch.zeros(num_atoms, s_i.shape[-1], 3).to(s_i.device)
    #     # s_i, v_i = self.embed_block(new_fea, nbrs=nbrs, r_ij=r_ij)

    #     for i, message_block in enumerate(self.message_blocks):
    #         update_block = self.update_blocks[i]
    #         ds_message, dv_message = message_block(
    #             s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
    #         )

    #         s_i = s_i + ds_message
    #         v_i = v_i + dv_message

    #         ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

    #         s_i = s_i + ds_update
    #         v_i = v_i + dv_update

    #     return s_i, xyz, r_ij, nbrs
