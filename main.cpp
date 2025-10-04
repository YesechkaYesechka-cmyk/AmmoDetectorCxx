#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>

// JSON (header-only): https://github.com/nlohmann/json
#include "nlohmann/json.hpp"

// TFLite
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/common.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// -------------------- Утилиты --------------------
struct Meta {
    std::vector<std::string> class_names;
    std::string base_model;   // "vgg16" | "resnet50" | "efficientnet"
    std::string preprocess;   // тот же маркер, что и base_model
    int img_h = 224;
    int img_w = 224;
};

Meta load_meta(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("Не удалось открыть classes.json: " + json_path);
    json j; in >> j;

    Meta m;
    if (!j.contains("class_names")) throw std::runtime_error("classes.json: нет поля class_names");
    for (auto& v : j["class_names"]) m.class_names.push_back(v.get<std::string>());

    if (j.contains("base_model"))  m.base_model  = j["base_model"].get<std::string>();
    if (j.contains("preprocess"))  m.preprocess  = j["preprocess"].get<std::string>();
    if (j.contains("img_height"))  m.img_h       = j["img_height"].get<int>();
    if (j.contains("img_width"))   m.img_w       = j["img_width"].get<int>();

    // унифицируем ключ (на всякий случай)
    std::transform(m.preprocess.begin(), m.preprocess.end(), m.preprocess.begin(), ::tolower);
    std::transform(m.base_model.begin(), m.base_model.end(), m.base_model.begin(), ::tolower);
    return m;
}

std::vector<std::string> gather_images(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            files.push_back(p.path().string());
        }
    }
    return files;
}

// -------------------- resize_with_pad --------------------
cv::Mat resize_with_pad_bgr(const cv::Mat& img_bgr, int target_w, int target_h) {
    double scale = std::min(
            static_cast<double>(target_w) / img_bgr.cols,
            static_cast<double>(target_h) / img_bgr.rows
    );
    int new_w = std::max(1, static_cast<int>(std::round(img_bgr.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(img_bgr.rows * scale)));

    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    int x = (target_w - new_w) / 2;
    int y = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, new_w, new_h)));
    return canvas;
}

// -------------------- Препроцессинги --------------------
// Средние ImageNet (BGR) для "caffe"-режима (VGG/ResNet в Keras):
static const float IMAGENET_MEAN_B = 103.939f;
static const float IMAGENET_MEAN_G = 116.779f;
static const float IMAGENET_MEAN_R = 123.680f;

// VGG16/ResNet50 (кафе-подобный): вход BGR float32, вычитание средних по каналам
void preprocess_vgg_resnet_inplace_bgr(cv::Mat& bgr) {
    bgr.convertTo(bgr, CV_32FC3);
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    ch[0] = ch[0] - IMAGENET_MEAN_B;
    ch[1] = ch[1] - IMAGENET_MEAN_G;
    ch[2] = ch[2] - IMAGENET_MEAN_R;
    cv::merge(ch, bgr);
}

// EfficientNet (tf-стек): RGB float32 в диапазоне [-1, 1]
void preprocess_efficientnet_inplace_rgb(cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0); // x/127.5 - 1
    bgr = rgb; // переиспользуем контейнер
}

// Обёртка по строковому имени препроцессинга
enum class PrepKind { VGG16, RESNET50, EFFICIENTNET };

PrepKind select_prep(const std::string& marker) {
    if (marker == "vgg16") return PrepKind::VGG16;
    if (marker == "resnet50") return PrepKind::RESNET50;
    if (marker == "efficientnet") return PrepKind::EFFICIENTNET;
    // По умолчанию пробуем как VGG/ResNet
    return PrepKind::VGG16;
}

// -------------------- Загрузка модели TFLite --------------------
std::unique_ptr<tflite::Interpreter> load_tflite(
        const std::string& model_path,
        std::unique_ptr<tflite::FlatBufferModel>& model_holder
) {
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

    interpreter->SetNumThreads(2);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("AllocateTensors() failed");
    }
    return interpreter;
}

// -------------------- Инференс одного изображения --------------------
struct Pred {
    std::string path;
    std::string label;
    float confidence;
    int class_index;
    std::vector<float> probs; // полный softmax
};

Pred predict_one(tflite::Interpreter* interp,
                 const std::string& path,
                 const Meta& meta,
                 PrepKind prep_kind)
{
    cv::Mat img_bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        throw std::runtime_error("Не удалось прочитать изображение: " + path);
    }

    // 1) resize_with_pad -> (H,W)
    cv::Mat padded = resize_with_pad_bgr(img_bgr, meta.img_w, meta.img_h);

    // 2) препроцессинг на месте
    switch (prep_kind) {
        case PrepKind::VGG16:
        case PrepKind::RESNET50:
            preprocess_vgg_resnet_inplace_bgr(padded);
            break;
        case PrepKind::EFFICIENTNET:
            preprocess_efficientnet_inplace_rgb(padded);
            break;
        default:
            preprocess_vgg_resnet_inplace_bgr(padded);
    }

    // 3) записываем в входной тензор [1,H,W,3] (NHWC, float32)
    float* input = interp->typed_input_tensor<float>(0);
    const size_t bytes = static_cast<size_t>(meta.img_w) * meta.img_h * 3 * sizeof(float);
    if (padded.type() != CV_32FC3) {
        throw std::runtime_error("Внутренняя ошибка: ожидался CV_32FC3 после препроцессинга");
    }
    std::memcpy(input, padded.data, bytes);

    // 4) run
    if (interp->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Invoke() failed");
    }

    // 5) читаем выход softmax [1, NUM_CLASSES]
    const float* out = interp->typed_output_tensor<float>(0);
    int out_idx = interp->outputs()[0];
    TfLiteTensor* out_tensor = interp->tensor(out_idx);
    if (out_tensor->dims->size != 2 || out_tensor->dims->data[0] != 1) {
        throw std::runtime_error("Неожиданная форма выхода: ожидается [1, NUM_CLASSES]");
    }
    int num_classes = out_tensor->dims->data[1];

    std::vector<float> probs(num_classes);
    for (int i = 0; i < num_classes; ++i) probs[i] = out[i];

    // argmax
    int best = 0;
    for (int i = 1; i < num_classes; ++i) if (probs[i] > probs[best]) best = i;

    Pred pred;
    pred.path = path;
    pred.class_index = best;
    pred.label = (best < (int)meta.class_names.size()) ? meta.class_names[best] : ("class_" + std::to_string(best));
    pred.confidence = probs[best];
    pred.probs = std::move(probs);
    return pred;
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: infer_multiclass <model.tflite> <classes.json> <images_dir>\n";
        return 1;
    }
    std::string model_path  = argv[1];
    std::string classes_json = argv[2];
    std::string images_dir  = argv[3];

    try {
        Meta meta = load_meta(classes_json);
        auto prep_kind = select_prep(!meta.preprocess.empty() ? meta.preprocess : meta.base_model);

        std::unique_ptr<tflite::FlatBufferModel> model_holder;
        auto interpreter = load_tflite(model_path, model_holder);

        auto images = gather_images(images_dir);
        if (images.empty()) {
            std::cout << "В директории нет изображений\n";
            return 0;
        }

        std::cout << "Модель: " << model_path << "\n";
        std::cout << "Классов: " << meta.class_names.size() << "  (";
        for (size_t i = 0; i < meta.class_names.size(); ++i) {
            std::cout << meta.class_names[i] << (i + 1 == meta.class_names.size() ? "" : ", ");
        }
        std::cout << ")\n";
        std::cout << "Препроцессинг: " << (meta.preprocess.empty() ? meta.base_model : meta.preprocess) << "\n";
        std::cout << "Размер: " << meta.img_w << "x" << meta.img_h << "\n";
        std::cout << "Найдено " << images.size() << " изображений\n";
        std::cout << "--------------------------------------------------\n";

        std::vector<int> hist(meta.class_names.size(), 0);
        for (const auto& p : images) {
            auto pr = predict_one(interpreter.get(), p, meta, prep_kind);
            std::string name = fs::path(p).filename().string();
            if (name.size() < 30) name += std::string(30 - name.size(), ' ');

            std::cout << name << " -> " << pr.label
                      << " (" << std::fixed << std::setprecision(2) << pr.confidence * 100.0f << "%)\n";

            if (pr.class_index >= 0 && pr.class_index < (int)hist.size()) hist[pr.class_index]++;
        }

        std::cout << "--------------------------------------------------\n";
        for (size_t i = 0; i < meta.class_names.size(); ++i) {
            std::cout << std::setw(20) << std::left << meta.class_names[i] << ": " << hist[i] << "\n";
        }
        std::cout << "Общее количество: " << images.size() << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}
