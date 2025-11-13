#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <random>
#include <atomic>
#include <algorithm> // For std::min_element (or custom min find)
#include <limits>    // For std::numeric_limits

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// --- (The BackgroundLoader class and ProcessedItem struct remain unchanged) ---
constexpr int TARGET_HEIGHT = 128;
constexpr int TARGET_WIDTH = 128;
constexpr int TARGET_CHANNELS = 4; // Changed from 3 to 4
constexpr size_t IMAGE_SIZE_FLOATS = TARGET_CHANNELS * TARGET_HEIGHT * TARGET_WIDTH;
constexpr size_t IMAGE_SIZE_BYTES = IMAGE_SIZE_FLOATS * sizeof(float);

struct ProcessedItem {
    std::unique_ptr<float[]> image_data;
    size_t dataset_idx;
    size_t image_idx;
};

class BackgroundLoader {
public:
    ~BackgroundLoader() {
        stop();
    }
    void set_data(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& jpeg_datasets, const std::vector<double>& dataset_probs) {
        m_jpeg_datasets = jpeg_datasets;
        m_dataset_probs = dataset_probs;

        // --- MODIFICATION: Initialize the return counts structure ---
        {
            std::lock_guard<std::mutex> lock(m_mtx_counts);
            m_image_return_counts.clear();
            m_image_return_counts.reserve(jpeg_datasets.size());
            for (const auto& dataset : jpeg_datasets) {
                m_image_return_counts.emplace_back(dataset.size(), 0); // Initialize all counts to 0
            }
        }
        // --- END MODIFICATION ---

        // m_current_indices is removed.

        m_dataset_dist = std::make_unique<std::discrete_distribution<>>(m_dataset_probs.begin(), m_dataset_probs.end());
    }
    void start(int num_threads, int max_queue_size) {
        if (m_running) return;
        m_max_deque_size = max_queue_size;
        m_running = true;
        for (int i = 0; i < num_threads; ++i) {
            m_threads.emplace_back(&BackgroundLoader::worker_loop, this);
        }
    }
    void clear_data() {
        // This method is called when threads are already stopped.
        m_jpeg_datasets.clear();
        m_dataset_probs.clear();
        
        // --- MODIFICATION: Clear the return counts ---
        {
            std::lock_guard<std::mutex> lock(m_mtx_counts);
            m_image_return_counts.clear();
        }
        // --- END MODIFICATION ---

        m_dataset_dist.reset();
    }
    void stop() {
        if (!m_running) return;
        m_running = false;
        m_cv_deque_not_full.notify_all();
        m_cv_deque_not_empty.notify_all();
        for (auto& t : m_threads) {
            if (t.joinable()) t.join();
        }
        m_threads.clear();
        
        {
            std::lock_guard<std::mutex> lock(m_mtx_deque);
            m_deque.clear();
        }
    }
    void fill_batch(PyArrayObject* image_array, PyArrayObject* index_array) {
        float* image_batch_ptr = (float*)PyArray_DATA(image_array);
        int64_t* index_batch_ptr = (int64_t*)PyArray_DATA(index_array);
        npy_intp batch_size = PyArray_DIMS(image_array)[0];

        for (npy_intp i = 0; i < batch_size; ++i) {
            std::unique_lock<std::mutex> lock(m_mtx_deque);

            m_cv_deque_not_empty.wait(lock, [this] {
                return !m_deque.empty() || !m_running;
            });

            if (!m_running && m_deque.empty()) return;

            ProcessedItem item = std::move(m_deque.front());
            m_deque.pop_front();

            lock.unlock();
            m_cv_deque_not_full.notify_one();

            float* dest_image_ptr = image_batch_ptr + (i * IMAGE_SIZE_FLOATS);
            memcpy(dest_image_ptr, item.image_data.get(), IMAGE_SIZE_BYTES);

            int64_t* dest_index_ptr = index_batch_ptr + (i * 2);
            dest_index_ptr[0] = item.dataset_idx;
            dest_index_ptr[1] = item.image_idx;
        }
    }
private:
    void worker_loop() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);

        while (m_running) {
            {
                std::unique_lock<std::mutex> lock(m_mtx_deque);
                m_cv_deque_not_full.wait(lock, [this] {
                    return m_deque.size() < m_max_deque_size || !m_running;
                });
                if (!m_running) break;
            }

            size_t dataset_idx = (*m_dataset_dist)(gen);

            if (m_jpeg_datasets[dataset_idx].empty()) {
                continue;
            }
            
            size_t image_idx;
            
            // --- MODIFICATION: New least-used-first selection logic ---
            {
                std::lock_guard<std::mutex> lock(m_mtx_counts);

                const auto& counts = m_image_return_counts[dataset_idx];
                size_t dataset_size = counts.size();
                size_t min_count = std::numeric_limits<size_t>::max();
                
                // 1. Find the minimum count
                for (size_t count : counts) {
                    if (count < min_count) {
                        min_count = count;
                    }
                }

                // 2. Collect all indices with the minimum count
                std::vector<size_t> least_used_indices;
                for (size_t i = 0; i < dataset_size; ++i) {
                    if (counts[i] == min_count) {
                        least_used_indices.push_back(i);
                    }
                }

                // 3. Randomly select from the least-used indices
                // Note: The least_used_indices vector must not be empty since min_count was found.
                std::uniform_int_distribution<> index_dist(0, least_used_indices.size() - 1);
                image_idx = least_used_indices[index_dist(gen)];

                // 4. Increment the count
                m_image_return_counts[dataset_idx][image_idx]++;
            }
            // --- END MODIFICATION ---

            // size_t image_idx = m_current_indices[dataset_idx]->fetch_add(1) % m_jpeg_datasets[dataset_idx].size(); // REMOVED
            const auto& jpeg_group = m_jpeg_datasets[dataset_idx][image_idx];

            if (jpeg_group.size() != TARGET_CHANNELS) {
                continue; 
            }
            
            std::vector<cv::Mat> single_channel_images;
            bool decode_success = true;
            for(const auto& jpeg_bytes : jpeg_group) {
                cv::Mat raw_data_mat(1, jpeg_bytes.size(), CV_8UC1, (void*)jpeg_bytes.data());
                cv::Mat decoded_image = cv::imdecode(raw_data_mat, cv::IMREAD_COLOR);
                if (decoded_image.empty()) {
                    decode_success = false;
                    break;
                }

                cv::Mat resized_image;
                cv::resize(decoded_image, resized_image, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

                cv::Mat gray_image;
                if (resized_image.channels() == 3) {
                    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);
                } else {
                    gray_image = resized_image;
                }

                cv::Mat equalized_image;
                cv::equalizeHist(gray_image, equalized_image);
                single_channel_images.push_back(equalized_image);
            }

            if (!decode_success || single_channel_images.size() != TARGET_CHANNELS) continue;

            cv::Mat merged_image;
            cv::merge(single_channel_images, merged_image);
            
            cv::Mat float_image;
            merged_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

            if (prob_dist(gen) < 0.3) {
                const float max_shift = 22.0f;
                const float max_rotation = 12.0f;
                const float max_scale = 0.1f;

                std::uniform_real_distribution<> rot_dist(-max_rotation, max_rotation);
                std::uniform_real_distribution<> scale_dist(-max_scale, max_scale);
                std::uniform_real_distribution<> shift_dist(-max_shift, max_shift);

                double angle = rot_dist(gen);
                double scale = 1.0 + scale_dist(gen);
                double shift_x = shift_dist(gen);
                double shift_y = shift_dist(gen);

                cv::Point2f center(float_image.cols / 2.0, float_image.rows / 2.0);
                cv::Mat transform_matrix = cv::getRotationMatrix2D(center, angle, scale);
                transform_matrix.at<double>(0, 2) += shift_x;
                transform_matrix.at<double>(1, 2) += shift_y;

                cv::warpAffine(float_image, float_image, transform_matrix, float_image.size(),
                            cv::INTER_LINEAR, cv::BORDER_REFLECT);
            }
            if (prob_dist(gen) < 0.4) {
                const float brightness_range = 0.2f;
                const float contrast_range = 0.6f;

                std::uniform_real_distribution<> bright_dist(-brightness_range, brightness_range);
                std::uniform_real_distribution<> contrast_dist(-contrast_range, contrast_range);

                double brightness = bright_dist(gen);
                double contrast = 1.0 + contrast_dist(gen);

                float_image.convertTo(float_image, -1, contrast, brightness);

                double minVal, maxVal;
                cv::minMaxLoc(float_image, &minVal, &maxVal);
                if (maxVal > 0) {
                    float_image /= maxVal;
                }
            }
            if (prob_dist(gen) < 0.3) {
                if (prob_dist(gen) < 0.5) {
                    const int max_kernel_size = 15;
                    std::uniform_int_distribution<> kernel_dist(1, max_kernel_size / 2);
                    int kernel_size = 2 * kernel_dist(gen) + 1;

                    std::uniform_real_distribution<> sigma_dist(0.1, 2.0);
                    double sigma = sigma_dist(gen);

                    cv::GaussianBlur(float_image, float_image, cv::Size(kernel_size, kernel_size), sigma);
                }
            }
            auto output_buffer = std::make_unique<float[]>(IMAGE_SIZE_FLOATS);
            float* dest = output_buffer.get();
            std::vector<cv::Mat> planes(TARGET_CHANNELS);
            cv::split(float_image, planes);
            const size_t plane_size_bytes = (TARGET_WIDTH * TARGET_HEIGHT) * sizeof(float);
            for (int i = 0; i < TARGET_CHANNELS; ++i) {
                memcpy(dest, planes[i].data, plane_size_bytes);
                dest += (plane_size_bytes / sizeof(float));
            }
            ProcessedItem item;
            item.image_data = std::move(output_buffer);
            item.dataset_idx = dataset_idx;
            item.image_idx = image_idx;

            {
                std::lock_guard<std::mutex> lock(m_mtx_deque);
                m_deque.push_back(std::move(item));
            }
            m_cv_deque_not_empty.notify_one();
        }
    }
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> m_jpeg_datasets;
    std::vector<double> m_dataset_probs;
    // --- MODIFICATION: Removed m_current_indices ---
    // std::vector<std::unique_ptr<std::atomic<size_t>>> m_current_indices;
    // --- END MODIFICATION ---

    // --- MODIFICATION: New members for tracking counts ---
    std::vector<std::vector<size_t>> m_image_return_counts;
    std::mutex m_mtx_counts; // Protects m_image_return_counts
    // --- END MODIFICATION ---

    std::unique_ptr<std::discrete_distribution<>> m_dataset_dist;
    std::deque<ProcessedItem> m_deque;
    std::mutex m_mtx_deque;
    std::condition_variable m_cv_deque_not_full;
    std::condition_variable m_cv_deque_not_empty;
    size_t m_max_deque_size;
    std::vector<std::thread> m_threads;
    std::atomic<bool> m_running{false};
};

// ... (The rest of the Python wrapper code remains unchanged as the interface functions still use BabbleLoader methods) ...

// The Python wrapper code is omitted for brevity, but it remains the same,
// as the changes are internal to the BackgroundLoader class.
typedef struct {
    PyObject_HEAD
    BackgroundLoader *loader_instance;
} BabbleLoaderObject;


extern "C" {

// MODIFIED: Implementation of the deallocator (like __del__)
static void BabbleLoader_dealloc(BabbleLoaderObject* self) {
    if (self->loader_instance) {
        // Stop() is implicitly called by the destructor of BackgroundLoader
        delete self->loader_instance;
        self->loader_instance = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// MODIFIED: Implementation of the constructor (like __new__)
static PyObject* BabbleLoader_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BabbleLoaderObject *self;
    self = (BabbleLoaderObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->loader_instance = nullptr;
    }
    return (PyObject*)self;
}

// MODIFIED: Implementation of the initializer (like __init__)
static int BabbleLoader_init(BabbleLoaderObject *self, PyObject *args, PyObject *kwds) {
    PyObject* jpeg_datasets_list;
    PyObject* dataset_probs_list;
    // MODIFIED: Added keyword argument parsing for clarity in Python
    static char *kwlist[] = {"jpeg_datasets", "dataset_probs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!", kwlist, 
                                     &PyList_Type, &jpeg_datasets_list, 
                                     &PyList_Type, &dataset_probs_list)) {
        // Error already set by PyArg_ParseTupleAndKeywords
        return -1;
    }

    Py_ssize_t num_datasets = PyList_Size(jpeg_datasets_list);
    if (num_datasets != PyList_Size(dataset_probs_list)) {
        PyErr_SetString(PyExc_ValueError, "jpeg_datasets and dataset_probs must have the same length.");
        return -1;
    }

    // MODIFIED: Create a new loader instance for this object
    try {
        self->loader_instance = new BackgroundLoader();
    } catch (const std::bad_alloc&) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate BackgroundLoader.");
        return -1;
    }

    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> datasets_data;
    datasets_data.reserve(num_datasets);
    std::vector<double> probs_data;
    probs_data.reserve(num_datasets);

    for (Py_ssize_t i = 0; i < num_datasets; ++i) {
        PyObject* prob_item = PyList_GetItem(dataset_probs_list, i);
        if (!PyFloat_Check(prob_item)) {
            PyErr_SetString(PyExc_TypeError, "All items in dataset_probs must be floats.");
            delete self->loader_instance; // Clean up on failure
            self->loader_instance = nullptr;
            return -1;
        }
        probs_data.push_back(PyFloat_AsDouble(prob_item));

        PyObject* dataset_list = PyList_GetItem(jpeg_datasets_list, i);
        if (!PyList_Check(dataset_list)) {
            PyErr_SetString(PyExc_TypeError, "Each item in jpeg_datasets must be a list of image groups.");
            delete self->loader_instance; // Clean up on failure
            self->loader_instance = nullptr;
            return -1;
        }

        Py_ssize_t dataset_size = PyList_Size(dataset_list);
        std::vector<std::vector<std::vector<uint8_t>>> single_dataset;
        single_dataset.reserve(dataset_size);

        for (Py_ssize_t j = 0; j < dataset_size; ++j) {
            PyObject* image_group_list = PyList_GetItem(dataset_list, j);
             if (!PyList_Check(image_group_list) || PyList_Size(image_group_list) != TARGET_CHANNELS) {
                PyErr_SetString(PyExc_TypeError, "Each image group must be a list of 4 bytes objects.");
                delete self->loader_instance; // Clean up on failure
                self->loader_instance = nullptr;
                return -1;
            }
            
            std::vector<std::vector<uint8_t>> single_image_group;
            single_image_group.reserve(TARGET_CHANNELS);

            for (Py_ssize_t k = 0; k < PyList_Size(image_group_list); ++k) {
                PyObject* item = PyList_GetItem(image_group_list, k);
                 if (!PyBytes_Check(item)) {
                    PyErr_SetString(PyExc_TypeError, "All items in an image group list must be bytes objects.");
                    delete self->loader_instance; // Clean up on failure
                    self->loader_instance = nullptr;
                    return -1;
                }
                char* buffer = PyBytes_AsString(item);
                Py_ssize_t len = PyBytes_Size(item);
                single_image_group.emplace_back(buffer, buffer + len);
            }
            single_dataset.push_back(single_image_group);
        }
        datasets_data.push_back(single_dataset);
    }

    self->loader_instance->set_data(datasets_data, probs_data);
    return 0; // Success
}


// MODIFIED: This is now a method of the BabbleLoaderObject type
static PyObject* BabbleLoader_start(BabbleLoaderObject* self, PyObject* args) {
    int num_threads = 0;
    int max_queue_size = 0;
    if (!PyArg_ParseTuple(args, "ii", &num_threads, &max_queue_size)) {
        return NULL;
    }
    if (!self->loader_instance) {
        PyErr_SetString(PyExc_RuntimeError, "Loader not initialized properly.");
        return NULL;
    }
    self->loader_instance->start(num_threads, max_queue_size);
    Py_RETURN_NONE;
}

static PyObject* BabbleLoader_reset(BabbleLoaderObject* self, PyObject* args, PyObject* kwds) {
    if (!self->loader_instance) {
        PyErr_SetString(PyExc_RuntimeError, "Loader not initialized properly.");
        return NULL;
    }

    // 1. Stop the running threads
    self->loader_instance->stop();

    // 2. Clear the internal data
    self->loader_instance->clear_data();

    // 3. Parse new data from Python
    PyObject* jpeg_datasets_list;
    PyObject* dataset_probs_list;
    static char *kwlist[] = {"jpeg_datasets", "dataset_probs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!", kwlist, 
                                     &PyList_Type, &jpeg_datasets_list, 
                                     &PyList_Type, &dataset_probs_list)) {
        return NULL; // Correct return for PyObject*
    }

    // CORRECTED: Py_ssize_t instead of Py_sssize_t
    Py_ssize_t num_datasets = PyList_Size(jpeg_datasets_list);
    if (num_datasets != PyList_Size(dataset_probs_list)) {
        PyErr_SetString(PyExc_ValueError, "jpeg_datasets and dataset_probs must have the same length.");
        return NULL; // Correct return for PyObject*
    }

    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> datasets_data;
    datasets_data.reserve(num_datasets);
    std::vector<double> probs_data;
    probs_data.reserve(num_datasets);

    for (Py_ssize_t i = 0; i < num_datasets; ++i) {
        PyObject* prob_item = PyList_GetItem(dataset_probs_list, i);
        if (!PyFloat_Check(prob_item)) {
            PyErr_SetString(PyExc_TypeError, "All items in dataset_probs must be floats.");
            return NULL; // Correct return for PyObject*
        }
        probs_data.push_back(PyFloat_AsDouble(prob_item));

        PyObject* dataset_list = PyList_GetItem(jpeg_datasets_list, i);
        if (!PyList_Check(dataset_list)) {
            PyErr_SetString(PyExc_TypeError, "Each item in jpeg_datasets must be a list.");
            return NULL; // Correct return for PyObject*
        }

        Py_ssize_t dataset_size = PyList_Size(dataset_list);
        std::vector<std::vector<std::vector<uint8_t>>> single_dataset;
        single_dataset.reserve(dataset_size);

        for (Py_ssize_t j = 0; j < dataset_size; ++j) {
            PyObject* image_group_list = PyList_GetItem(dataset_list, j);
            if (!PyList_Check(image_group_list) || PyList_Size(image_group_list) != TARGET_CHANNELS) {
                PyErr_SetString(PyExc_TypeError, "Each image group must be a list of 4 bytes objects.");
                return NULL; // Correct return for PyObject*
            }
            
            std::vector<std::vector<uint8_t>> single_image_group;
            single_image_group.reserve(TARGET_CHANNELS);

            for (Py_ssize_t k = 0; k < PyList_Size(image_group_list); ++k) {
                PyObject* item = PyList_GetItem(image_group_list, k);
                 if (!PyBytes_Check(item)) {
                    PyErr_SetString(PyExc_TypeError, "All items in an image group list must be bytes objects.");
                    return NULL; // Correct return for PyObject*
                }
                char* buffer = PyBytes_AsString(item);
                Py_ssize_t len = PyBytes_Size(item);
                single_image_group.emplace_back(buffer, buffer + len);
            }
            single_dataset.push_back(single_image_group);
        }
        datasets_data.push_back(single_dataset);
    }
    
    // 4. Set the new data. This will reuse vector memory if possible.
    self->loader_instance->set_data(datasets_data, probs_data);

    Py_RETURN_NONE; // Correct return for success
}

// MODIFIED: This is now a method to explicitly stop. It's also called by the deallocator.
static PyObject* BabbleLoader_stop(BabbleLoaderObject* self, PyObject* args) {
    if (self->loader_instance) {
        self->loader_instance->stop();
    }
    Py_RETURN_NONE;
}

// MODIFIED: This is now a method of the BabbleLoaderObject type
static PyObject* BabbleLoader_fill_batch(BabbleLoaderObject* self, PyObject* args) {
    PyObject* image_array_obj = NULL;
    PyObject* index_array_obj = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image_array_obj, &PyArray_Type, &index_array_obj)) {
        return NULL;
    }

    PyArrayObject* image_array = (PyArrayObject*)image_array_obj;
    if (PyArray_NDIM(image_array) != 4 || PyArray_DIMS(image_array)[1] != TARGET_CHANNELS ||
        PyArray_DIMS(image_array)[2] != TARGET_HEIGHT || PyArray_DIMS(image_array)[3] != TARGET_WIDTH) {
        PyErr_SetString(PyExc_ValueError, "Image array has incorrect dimensions. Expected (batch, 4, 128, 128).");
        return NULL;
    }
    if (PyArray_TYPE(image_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "Image array must be of type float32.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(image_array)) {
        PyErr_SetString(PyExc_ValueError, "Image array must be C-contiguous.");
        return NULL;
    }

    PyArrayObject* index_array = (PyArrayObject*)index_array_obj;
    if (PyArray_NDIM(index_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Index array must be 2-dimensional.");
        return NULL;
    }
    if (PyArray_DIMS(image_array)[0] != PyArray_DIMS(index_array)[0]) {
        PyErr_SetString(PyExc_ValueError, "Batch size of image array and index array must match.");
        return NULL;
    }
    if (PyArray_DIMS(index_array)[1] != 2) {
        PyErr_SetString(PyExc_ValueError, "The second dimension of the index array must be 2.");
        return NULL;
    }
    if (PyArray_TYPE(index_array) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "Index array must be of type int64.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(index_array)) {
        PyErr_SetString(PyExc_ValueError, "Index array must be C-contiguous.");
        return NULL;
    }

    if (!self->loader_instance) {
        PyErr_SetString(PyExc_RuntimeError, "Loader has not been initialized.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    self->loader_instance->fill_batch(image_array, index_array);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// MODIFIED: Define the methods for our new Python type
static PyMethodDef BabbleLoader_methods[] = {
    {"start", (PyCFunction)BabbleLoader_start, METH_VARARGS, "Start the background worker threads."},
    {"stop", (PyCFunction)BabbleLoader_stop, METH_NOARGS, "Stop the workers and clean up. This is called automatically when the object is destroyed."},
    {"reset", (PyCFunction)BabbleLoader_reset, METH_VARARGS | METH_KEYWORDS, "Stop, clear, and reload with new data."},
    {"fill_batch", (PyCFunction)BabbleLoader_fill_batch, METH_VARARGS, "Fill pre-allocated NumPy arrays with a batch of images and their corresponding indices."},
    {NULL}  /* Sentinel */
};

// MODIFIED: Define the new Python type itself
static PyTypeObject BabbleLoaderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "babble_data.Loader",                       /* tp_name */
    sizeof(BabbleLoaderObject),                 /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)BabbleLoader_dealloc,           /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "A multi-threaded data loader for image batches.", /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    BabbleLoader_methods,                       /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)BabbleLoader_init,                /* tp_init */
    0,                                          /* tp_alloc */
    BabbleLoader_new,                           /* tp_new */
};


// MODIFIED: The module definition no longer has top-level functions
static PyMethodDef BabbleDataMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef babble_data_module = {
    PyModuleDef_HEAD_INIT, "babble_data", "A multi-threaded data loading module.", -1, BabbleDataMethods
};

PyMODINIT_FUNC PyInit_babble_data(void) {
    PyObject *m;
    
    // MODIFIED: Add our new type to the module during initialization
    if (PyType_Ready(&BabbleLoaderType) < 0)
        return NULL;

    m = PyModule_Create(&babble_data_module);
    if (m == NULL)
        return NULL;

    import_array(); // Important: Initialize NumPy
    if (PyErr_Occurred()) {
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&BabbleLoaderType);
    if (PyModule_AddObject(m, "Loader", (PyObject *)&BabbleLoaderType) < 0) {
        Py_DECREF(&BabbleLoaderType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

} // extern "C"
