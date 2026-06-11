import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api'

function getHeaders() {
  return {
    'Content-Type': 'application/json'
  }
}

export async function processFilesForRAG(files) {
  const headers = getHeaders()
  const response = await axios.post(`${API_BASE_URL}/rag/upload`, { files }, { headers })
  return response.data
}

export async function readLocalFile(filePath) {
  const headers = getHeaders()
  const response = await axios.post(`${API_BASE_URL}/rag/read-file`, {}, {
    headers,
    params: { file_path: filePath }
  })
  return response.data
}

export async function getRAGFiles() {
  const headers = getHeaders()
  const response = await axios.get(`${API_BASE_URL}/rag/files`, { headers })
  return response.data
}

export async function deleteRAGFile(fileId) {
  const headers = getHeaders()
  const response = await axios.delete(`${API_BASE_URL}/rag/files/${fileId}`, { headers })
  return response.data
}

export async function retrieveDocuments(query, topK = 5) {
  const headers = getHeaders()
  const response = await axios.post(
    `${API_BASE_URL}/rag/retrieve`,
    {},
    {
      headers,
      params: { query, top_k: topK }
    }
  )
  return response.data
}