cmake_minimum_required(VERSION 3.23)
project(cats_dogs_infer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== OpenCV =====
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ВАЖНО: задать версию TF для core/public/* ---
add_compile_definitions(
        TF_MAJOR_VERSION=2
        TF_MINOR_VERSION=20
        TF_PATCH_VERSION=0
        TF_VERSION_SUFFIX=\"\"
)

# ===== Общие опции TensorFlow Lite =====
# Под Windows/MSVC обычно проще отключить XNNPACK.
# Если очень нужно XNNPACK — можно попробовать включить позже, но начните с OFF.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK_DELEGATE ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU    OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_NNAPI  OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_MMAP   OFF CACHE BOOL "" FORCE)

# ===== Специфика компилятора =====
if (MSVC)
    message(STATUS "Configuring for MSVC")

    # Большие объектные файлы и параллельная сборка
    add_compile_options(/bigobj /MP /EHsc)

    # Чуть более строгий стандарт, корректное значение __cplusplus
    add_compile_options(/Zc:__cplusplus)

    # Убираем конфликтующие макросы Windows
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)

    # Рантайм (/MD по умолчанию). Если хотите статический рантайм — смените на MultiThreaded*
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.15")
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL") # /MD
        # альтернативы:
        # "MultiThreaded"      # /MT
        # "MultiThreadedDebug" # /MTd
        # "MultiThreadedDebugDLL" # /MDd
    endif()
else()
    message(STATUS "Non-MSVC toolchain detected")
    # На всякий случай удалим ваши MinGW-ключи, если остались в проекте
    foreach(lang C CXX)
        string(REGEX REPLACE "-mpreferred-stack-boundary=[0-9]+" "" CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS}")
    endforeach()
endif()

# ===== Подключаем TensorFlow Lite =====
# ВАЖНО: путь должен указывать на директорию lite CMake (tensorflow/tensorflow/lite)
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/deps/tensorflow/tensorflow/lite")

# Создаём отдельную директорию для сборки TFLite, чтобы не пачкать корневой build
# и не тянуть цели в ALL:
add_subdirectory(${TFLITE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/build_lite EXCLUDE_FROM_ALL)

if (MSVC)
    # Разрешаем макросизацию ключевых слов в MSVC STL (иначе xkeycheck.h даст C1189)
    target_compile_definitions(tensorflow-lite PRIVATE _ALLOW_KEYWORD_MACROS)
    # Делаем приватные поля публичными ТОЛЬКО внутри сборки tflite (обход бага 2.20.0)
    target_compile_definitions(tensorflow-lite PRIVATE private=public protected=public)
endif()

# ===== Ваш бинарник =====
add_executable(AmmoDetector main.cpp)
target_include_directories(AmmoDetector PRIVATE ${OpenCV_INCLUDE_DIRS})

# Линкуем статическую tflite (цель называется tensorflow-lite внутри lite/CMakeLists)
target_link_libraries(AmmoDetector PRIVATE ${OpenCV_LIBS} tensorflow-lite)

# Если понадобятся дополнительные дефайны для Windows-сокетов/файлов, раскомментируйте:
# target_compile_definitions(AmmoDetector PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
