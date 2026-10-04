import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const restoreImage = async (file, corruptionType, params) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('corruption_type', corruptionType);
  formData.append('corruption_params', JSON.stringify(params));

  const response = await axios.post(`${API_URL}/universal-restore`, formData);
  return response.data;
};

export const routeImage = async (file, corruptionType, params) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('corruption_type', corruptionType);
  formData.append('corruption_params', JSON.stringify(params));

  const response = await axios.post(`${API_URL}/hard-route`, formData);
  return response.data;
};

export const softMoE = async (file, corruptionType, params) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('corruption_type', corruptionType);
  formData.append('corruption_params', JSON.stringify(params));

  const response = await axios.post(`${API_URL}/soft-mixture`, formData);
  return response.data;
};

export const generateSketch = async (file, styleIdx) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('style_idx', styleIdx);

  const response = await axios.post(`${API_URL}/face-to-sketch`, formData);
  return response.data;
};
