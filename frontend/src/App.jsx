import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [jobId, setJobId] = useState('')
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isUploading, setIsUploading] = useState(false)

  const fileInputRef = useRef(null)
  const consecutive404Ref = useRef(0)
  const consecutiveNetworkErrorRef = useRef(0)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError('')
      setJobId('')
      setStatus('')
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.')
      return
    }

    // Basic validation (500MB max)
    if (file.size > 500 * 1024 * 1024) {
      setError('File size exceeds the 500MB limit.')
      return
    }

    setIsUploading(true)
    setError('')
    setJobId('')
    setStatus('uploading')
    setProgress('Uploading file...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${baseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`)
      }

      const data = await response.json()
      setJobId(data.job_id)
      setStatus(data.status)
      setProgress('Upload complete, queued for processing.')
      consecutive404Ref.current = 0
      consecutiveNetworkErrorRef.current = 0
    } catch (err) {
      setError(err.message)
      setStatus('error')
    } finally {
      setIsUploading(false)
    }
  }

  // Poll for status
  useEffect(() => {
    let intervalId

    const checkStatus = async () => {
      if (!jobId || status === 'done' || status === 'failed' || status === 'error') {
        return
      }

      try {
        const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
        const response = await fetch(`${baseUrl}/api/jobs/${jobId}`)
        if (response.status === 404) {
          consecutive404Ref.current += 1
          consecutiveNetworkErrorRef.current = 0
          if (consecutive404Ref.current >= 3) {
            setError('Job no longer available. The server may have restarted during processing. Please try again.')
            setStatus('error')
          }
          return
        }
        if (!response.ok) throw new Error('Failed to fetch job status')

        consecutive404Ref.current = 0
        consecutiveNetworkErrorRef.current = 0
        const data = await response.json()
        setStatus(data.status)
        setProgress(data.progress || '')

        if (data.status === 'done') {
          // Fetch result
          const resResponse = await fetch(`${baseUrl}/api/jobs/${jobId}/result`)
          if (resResponse.ok) {
            const resData = await resResponse.json()
            setResult(resData)
          }
        } else if (data.status === 'failed') {
          setError(data.error || 'Processing failed on the server')
        }
      } catch (err) {
        console.error('Polling error:', err)
        consecutiveNetworkErrorRef.current += 1
        consecutive404Ref.current = 0
        if (consecutiveNetworkErrorRef.current >= 3) {
          setError('Connection lost. Please check if the server is running and try again.')
          setStatus('error')
        }
      }
    }

    if (jobId && status !== 'done' && status !== 'failed' && status !== 'error') {
      intervalId = setInterval(checkStatus, 2000)
    }

    return () => clearInterval(intervalId)
  }, [jobId, status])

  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <h1>Whisper Diarization</h1>
          <p>Local Speech-to-Text with Speaker Identification</p>
        </div>
      </header>

      <main className="main-content">
        <div className="card upload-card">
          <h2>Upload Audio or Video</h2>
          <div
            className={`file-drop-area ${file ? 'has-file' : ''}`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="audio/*,video/*"
              className="hidden-input"
            />
            {file ? (
              <div className="file-info">
                <span className="file-icon">🎵</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon">📁</span>
                <p>Click or drag file to upload</p>
                <span className="upload-hints">Supports mp3, wav, mp4, etc. (Max 500MB)</span>
              </div>
            )}
          </div>

          <button
            className="btn-primary"
            onClick={handleUpload}
            disabled={!file || isUploading || ['queued', 'running'].includes(status)}
          >
            {isUploading ? 'Uploading...' : 'Process Audio'}
          </button>
        </div>

        {error && (
          <div className="alert error-alert">
            <span className="alert-icon">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {(status || jobId) && status !== 'error' && (
          <div className="card status-card">
            <div className="status-header">
              <h3>Processing Status</h3>
              <span className={`status-badge status-${status}`}>{status}</span>
            </div>

            <div className="progress-container">
              {['uploading', 'queued', 'running'].includes(status) && (
                <div className="loader-bar">
                  <div className="loader-fill"></div>
                </div>
              )}
              <p className="progress-text">{progress || 'Initializing...'}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="card result-card">
            <div className="result-header">
              <h2>Transcript</h2>
              <div className="download-actions">
                <a href={`${baseUrl}${result.download_txt}`} className="btn-secondary" download>
                  TXT
                </a>
                <a href={`${baseUrl}${result.download_json}`} className="btn-secondary" download>
                  JSON
                </a>
              </div>
            </div>

            <div className="transcript-container">
              {result.segments && result.segments.map((seg, idx) => {
                const start = new Date(seg.start * 1000).toISOString().substring(11, 19);
                return (
                  <div key={idx} className={`segment speaker-${seg.speaker.replace(/ /g, '-').toLowerCase()}`}>
                    <div className="segment-meta">
                      <span className="speaker-label">{seg.speaker}</span>
                      <span className="time-label">{start}</span>
                    </div>
                    <p className="segment-text">{seg.text}</p>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
