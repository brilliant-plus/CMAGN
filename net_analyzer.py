# -*- coding: utf-8 -*-
from xj_KATZ import KATZ, katz_similarity
from xj_WKNKN import WKNKN
from xj_NCP import NCP
from sim_process import jac_similarity, cos_similarity, GIP_similarity

def reconstruct_similarity_network(m_embeddings, c_embeddings, reconstruct_similarity):
    # Reconstructing similarity networks through embedding features
    # sim1: cos
    if reconstruct_similarity == "cos":
        m_similarity_network, c_similarity_network = cos_similarity(m_embeddings), cos_similarity(c_embeddings)
    # sim2: GIP
    if reconstruct_similarity == "GIP":
        m_similarity_network, c_similarity_network = GIP_similarity(m_embeddings), GIP_similarity(c_embeddings)
    # # sim3: jac
    if reconstruct_similarity == "jac":
        m_similarity_network, c_similarity_network = jac_similarity(m_embeddings), jac_similarity(c_embeddings)

    return m_similarity_network, c_similarity_network

def network_analyzer(m_similarity_network, c_similarity_network, new_AS, para_analyzer):
    # Get prediction matrix by using different network_analyzer

    # xj_NCP
    if para_analyzer == "xj-NCP":
        predictedMatrix = NCP(m_similarity_network, c_similarity_network, new_AS).network_NCP()

    # xj_KATZ
    if para_analyzer == "xj-KATZ":
        predictedMatrix = KATZ(m_similarity_network, c_similarity_network, new_AS).get_matrix()

    # xj_WKNKN
    if para_analyzer == "xj-WKNKN":
        predictedMatrix = WKNKN(m_similarity_network, c_similarity_network, new_AS).get_scores()

    return predictedMatrix