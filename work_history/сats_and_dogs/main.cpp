#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// TFLite
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/common.h"

namespace fs = std::filesystem;

// -------------------- Параметры модели --------------------
static const int IMG_W = 224;
static const int IMG_H = 224;
// Классы: 0=Кошка, 1=Собака
static const char* CLASS_CAT = "Кошка";
static const char* CLASS_DOG = "Собака";

// Средние для VGG16 (режим 'caffe') в порядке B, G, R:
static const float VGG_MEAN_B = 103.939f;
static const float VGG_MEAN_G = 116.779f;
static const float VGG_MEAN_R = 123.68f;

// -------------------- resize_with_pad --------------------
cv::Mat resize_with_pad_bgr(const cv::Mat& img_bgr, int target_w, int target_h) {
    // масштаб с сохранением пропорций
    double scale = std::min(
            static_cast<double>(target_w) / img_bgr.cols,
            static_cast<double>(target_h) / img_bgr.rows
    );
    int new_w = std::max(1, static_cast<int>(std::round(img_bgr.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(img_bgr.rows * scale)));

    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    // создаём холст target_w x target_h, чёрный фон (как у tf.image.resize_with_pad)
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    int x = (target_w - new_w) / 2;
    int y = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, new_w, new_h)));
    return canvas;
}

// -------------------- VGG16 preprocess_input (caffe) --------------------
// На вход: BGR в [0..255]. На выход: float32, BGR, с вычитанием средних.
void vgg16_preprocess_inplace(cv::Mat& img_bgr) {
    img_bgr.convertTo(img_bgr, CV_32FC3); // float32

    // Вычитаем средние в порядке каналов B, G, R
    std::vector<cv::Mat> ch(3);
    cv::split(img_bgr, ch);
    ch[0] = ch[0] - VGG_MEAN_B; // B
    ch[1] = ch[1] - VGG_MEAN_G; // G
    ch[2] = ch[2] - VGG_MEAN_R; // R
    cv::merge(ch, img_bgr);
}

// -------------------- Загрузка модели TFLite --------------------
std::unique_ptr<tflite::Interpreter> load_tflite(const std::string& model_path, std::unique_ptr<tflite::FlatBufferModel>& model_holder) {
    model_holder = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_holder) {
        throw std::runtime_error("Не удалось загрузить .tflite модель: " + model_path);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model_holder, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Не удалось создать интерпретатор TFLite");
    }

    // можно настроить число потоков
    interpreter->SetNumThreads(2);

    // В некоторых сборках полезно: динамически задать форму входа, но чаще уже [1,224,224,3]
    // interpreter->ResizeInputTensor(interpreter->inputs()[0], {1, IMG_H, IMG_W, 3});

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("AllocateTensors() failed");
    }

    return interpreter;
}

// -------------------- Предсказание для одного изображения --------------------
struct Prediction {
    std::string path;
    std::string label;
    float confidence; // max(p, 1-p)
    float raw;        // p из сигмоиды
};

Prediction predict_one(tflite::Interpreter* interp, const std::string& path) {
    // 1) читаем
    cv::Mat img_bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        throw std::runtime_error("Не удалось прочитать изображение: " + path);
    }

    // 2) resize_with_pad до 224x224
    cv::Mat padded = resize_with_pad_bgr(img_bgr, IMG_W, IMG_H);

    // 3) preprocess_input (VGG16, caffe): BGR float32 + вычитание средних
    vgg16_preprocess_inplace(padded); // теперь CV_32FC3

    // 4) записываем в входной тензор [1,224,224,3], float32, NHWC
    float* input = interp->typed_input_tensor<float>(0);
    // OpenCV хранит HWC, float32 → можно копировать плотно
    // Убедимся, что порядок каналов BGR сохраняем (так как обучали в 'caffe' режиме)
    // Keras VGG16: preprocess_input делает RGB->BGR внутри. Мы подали BGR и вычли BGR-средние,
    // это эквивалентно.
    std::memcpy(input, padded.data, IMG_W * IMG_H * 3 * sizeof(float));

    // 5) run
    if (interp->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Invoke() failed");
    }

    // 6) читаем выход: [1,1], сигмоида
    const float* out = interp->typed_output_tensor<float>(0);
    float p = out[0]; // вероятность класса "dog" (1), как в вашей модели

    Prediction pr;
    pr.path = path;
    pr.raw = p;
    pr.confidence = (p > 0.5f) ? p : (1.f - p);
    pr.label = (p > 0.5f) ? CLASS_DOG : CLASS_CAT;
    return pr;
}

// -------------------- Сбор всех изображений из директории --------------------
std::vector<std::string> gather_images(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        for (auto& c : ext) c = std::tolower(c);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            files.push_back(p.path().string());
        }
    }
    return files;
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: infer <cats_vs_dogs_model.tflite> <images_dir>\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string images_dir = argv[2];

    try {
        std::unique_ptr<tflite::FlatBufferModel> model_holder;
        auto interpreter = load_tflite(model_path, model_holder);

        auto images = gather_images(images_dir);
        if (images.empty()) {
            std::cout << "В директории нет изображений\n";
            return 0;
        }

        std::cout << "Найдено " << images.size() << " изображений для предсказания\n";
        std::cout << "--------------------------------------------------\n";

        int cats = 0, dogs = 0;
        for (const auto& p : images) {
            auto pr = predict_one(interpreter.get(), p);
            std::cout << fs::path(p).filename().string();
            if (fs::path(p).filename().string().size() < 20) {
                std::cout << std::string(20 - fs::path(p).filename().string().size(), ' ');
            }
            std::cout << " -> " << pr.label << " (" << std::fixed << std::setprecision(2) << pr.confidence*100.0f << "%)\n";
            if (pr.label == CLASS_DOG) ++dogs; else ++cats;
        }

        std::cout << "--------------------------------------------------\n";
        std::cout << "Статистика: Кошки - " << cats << ", Собаки - " << dogs << "\n";
        std::cout << "Общее количество: " << images.size() << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}
