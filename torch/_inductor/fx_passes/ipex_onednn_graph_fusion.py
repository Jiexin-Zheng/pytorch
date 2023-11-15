import logging
from torch.fx import Graph
from torch._inductor import config
from torch._inductor.fx_passes.onednn_graph_fusion import rewrite_graph, build_onednn_graph, fuse_graph, reapply_decomps

log = logging.getLogger(__name__)

def onednn_graph_fuse_fx(gm_graph: Graph, is_inference: bool=True):
    log.info("Compiling graph with oneDNN backend")
    gm = gm_graph.owning_module
    rewrite_graph(gm)
    log.debug("Build oneDNN graph")
    onednn_graph = build_onednn_graph(gm)
    onednn_graph.is_inference = is_inference
    log.debug("Fuse fx graph based on oneDNN graph partitions")
    fuse_graph(gm, onednn_graph)
    log.debug("Re-apply Inductor Decomps after fusion for any un-lowered ops")
    reapply_decomps(gm)
    log.info("Finished compiling graph with oneDNN backend")
    log.debug("====== Fx Graph after oneDNN compile ======")
    log.debug(gm.print_readable(print_output=False))
    config.cpp.onednn_graph = False
    print("turn off onednn graph temporarily")
    return gm

def ipex_post_pass_for_onednn_graph(gm_graph: Graph):
    config.cpp.onednn_graph = True
    print("turn on onednn graph back")