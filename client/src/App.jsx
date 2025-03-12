"use client"

import { useState } from "react"
import axios from "axios"
import {
  FileImage,
  Brain,
  Microscope,
  Dna,
  AlertCircle,
  Loader2,
  ChevronDown,
  Check
} from "lucide-react"
import "./App.css"

function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [dxType, setDxType] = useState("")
  const [age, setAge] = useState("")
  const [sex, setSex] = useState("")
  const [localization, setLocalization] = useState("")
  const [result, setResult] = useState(null)
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("form")

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setResult(null)

    if (!image) {
      setError("Please select an image before predicting.")
      return
    }

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append("image", image)
      formData.append("dx_type", dxType)
      formData.append("age", age)
      formData.append("sex", sex)
      formData.append("localization", localization)

      const response = await axios.post("https://skin-lesion-classifier-backend.vercel.app/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResult(response.data)
      setActiveTab("results")
    } catch (err) {
      setError("Error: " + (err.response?.data?.error || err.message))
    } finally {
      setIsLoading(false)
    }
  }

  const renderForm = () => (
    <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-2xl mx-auto">
      <div className="mb-6 text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Skin Lesion Image</h2>
        <p className="text-gray-600">Upload a clear image of the skin lesion for analysis</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Image Upload */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Lesion Image <span className="text-red-500">*</span>
          </label>
          <div
            className={`border-2 border-dashed rounded-lg p-4 transition-all ${
              imagePreview ? "border-green-400" : "border-gray-300 hover:border-primary"
            } cursor-pointer bg-gray-50`}
            onClick={() => document.getElementById("image-upload").click()}
          >
            <input id="image-upload" type="file" accept="image/*" onChange={handleImageChange} className="hidden" />

            {imagePreview ? (
              <div className="space-y-3">
                <div className="relative w-full h-48 rounded-lg overflow-hidden">
                  <img src={imagePreview || "/placeholder.svg"} alt="Preview" className="w-full h-full object-cover" />
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                    <p className="text-white text-sm truncate">{image?.name}</p>
                  </div>
                </div>
                <p className="text-sm text-center text-green-600 flex items-center justify-center">
                  <Check size={16} className="mr-1" /> Image uploaded successfully
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-6">
                <FileImage className="h-12 w-12 text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">Drag and drop or click to upload</p>
                <p className="text-xs text-gray-400 mt-1">PNG, JPG, JPEG up to 10MB</p>
              </div>
            )}
          </div>
        </div>

        {/* Patient Information */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <h3 className="font-medium text-gray-800 mb-3">Patient Information</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
              <input
                type="text"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="e.g., 45"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sex</label>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary appearance-none bg-white"
              >
                <option value="">Select</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Dx Type</label>
              <input
                type="text"
                value={dxType}
                onChange={(e) => setDxType(e.target.value)}
                placeholder="Diagnosis type"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Localization</label>
              <input
                type="text"
                value={localization}
                onChange={(e) => setLocalization(e.target.value)}
                placeholder="e.g., arm, face, back"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              />
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !image}
          className={`w-full flex items-center justify-center py-3 px-4 rounded-md  font-medium transition-all ${
            isLoading || !image ? "bg-gray-400 cursor-not-allowed" : "bg-primary hover:bg-primary/90"
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="animate-spin mr-2 h-5 w-5" />
              Processing...
            </>
          ) : (
            <>
              <Brain className="mr-2 h-5 w-5" />
              Analyze Lesion
            </>
          )}
        </button>
      </form>
    </div>
  )

  const renderResults = () => {
    if (!result) return null

    return (
      <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-4xl mx-auto animate-fadeIn">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
          <button
            onClick={() => setActiveTab("form")}
            className="text-sm text-primary hover:text-primary/80 flex items-center"
          >
            <ChevronDown className="h-4 w-4 mr-1 transform rotate-90" />
            Back to form
          </button>
        </div>

        {/* Main image */}
        <div className="mb-8 bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="md:w-1/3">
              <div className="rounded-lg overflow-hidden border border-gray-200">
                <img src={imagePreview || "/placeholder.svg"} alt="Original" className="w-full h-auto object-cover" />
              </div>
              <p className="text-sm text-center text-gray-500 mt-2">Original Image</p>
            </div>
            <div className="md:w-2/3">
              <div className="rounded-lg overflow-hidden border border-gray-200">
                <img
                  src={result.plot_image || "/placeholder.svg"}
                  alt="Analysis"
                  className="w-full h-auto object-cover"
                />
              </div>
              <p className="text-sm text-center text-gray-500 mt-2">Analysis Results</p>
            </div>
          </div>
        </div>

        {/* Results cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* CNN Output */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-5 border border-blue-100 shadow-sm">
            <div className="flex items-center mb-3">
              <Brain className="h-5 w-5 text-blue-600 mr-2" />
              <h3 className="font-bold text-blue-800">CNN Analysis</h3>
            </div>
            <p className="text-gray-700 mb-3">{result.cnn_output}</p>

            {result.all_class_probabilities && (
              <div className="mt-3 pt-3 border-t border-blue-100">
                <p className="text-sm font-medium text-blue-800 mb-2">Class Probabilities:</p>
                <div className="text-xs text-gray-600 space-y-1">
                  {result.all_class_probabilities.split(", ").map((prob, index) => {
                    const [className, value] = prob.split(": ")
                    const percentage = Number.parseFloat(value) * 100
                    return (
                      <div key={index} className="flex items-center">
                        <div className="w-24 font-medium">{className}:</div>
                        <div className="flex-1">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${percentage}%` }}></div>
                          </div>
                        </div>
                        <div className="w-16 text-right">{percentage.toFixed(1)}%</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          {/* NLP Output */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-5 border border-purple-100 shadow-sm">
            <div className="flex items-center mb-3">
              <Dna className="h-5 w-5 text-purple-600 mr-2" />
              <h3 className="font-bold text-purple-800">NLP Analysis</h3>
            </div>
            <p className="text-gray-700">{result.nlp_output}</p>
          </div>

          {/* Segmentation Output */}
          <div className="bg-gradient-to-br from-green-50 to-teal-50 rounded-xl p-5 border border-green-100 shadow-sm">
            <div className="flex items-center mb-3">
              <Microscope className="h-5 w-5 text-green-600 mr-2" />
              <h3 className="font-bold text-green-800">Segmentation Analysis</h3>
            </div>
            <p className="text-gray-700">{result.segmentation_output}</p>
          </div>

          {/* Final Output */}
          <div className="bg-gradient-to-br from-amber-50 to-yellow-50 rounded-xl p-5 border border-amber-100 shadow-sm">
            <div className="flex items-center mb-3">
              <AlertCircle className="h-5 w-5 text-amber-600 mr-2" />
              <h3 className="font-bold text-amber-800">Final Diagnosis</h3>
            </div>
            <p className="text-gray-700 font-medium">{result.final_output}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 ml-70">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <header className="text-center mb-10">
          <div className="inline-flex items-center justify-center p-2 bg-primary/10 rounded-full mb-4">
            <Microscope className="h-6 w-6 text-primary" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Skin Lesion Classifier</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Advanced deep learning system for skin lesion classification using image segmentation and NLP techniques
          </p>
        </header>

        {error && (
          <div className="max-w-2xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-start">
            <AlertCircle className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-sm w-full">
              <div className="flex flex-col items-center">
                <Loader2 className="h-12 w-12 text-primary animate-spin mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-1">Processing Image</h3>
                <p className="text-gray-500 text-center">
                  Our AI models are analyzing your image. This may take a moment...
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === "form" ? renderForm() : renderResults()}
      </div>
    </div>
  )
}

export default App
