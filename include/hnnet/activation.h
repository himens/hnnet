#pragma once
#include <any>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "hnnet/types.h"

namespace hnnet {
    template <typename T>
        concept ActivationType = requires (const T &activation, const real_t x) {
            { activation(x) } -> std::same_as<real_t>;
            { activation.derivative(x) } -> std::same_as<real_t>;
        };
    //////////////////////
    // Activation class //
    //////////////////////
    class Activation {
        public:
            // Store a copyable activation without depending on its concrete type
            template <typename T>
                requires (not std::same_as<std::remove_cvref_t<T>, Activation>) and ActivationType<T> and std::copy_constructible<T>
                explicit Activation(T activation) : _value(std::move(activation)), _vtable(&_vtable_for<T>) {}
            // Compute the activation value
            real_t operator()(const real_t x) const {
                return _vtable->call(_value, x);
            }
            // Compute the activation derivative
            real_t derivative(const real_t x) const {
                return _vtable->derive(_value, x);
            }
            // Retrieve the stored activation when its type matches
            template <typename T>
                const T* get_if() const {
                    return std::any_cast<T>(&_value);
                }
        private:
            struct VTable {
                real_t (*call)(const std::any&, real_t);
                real_t (*derive)(const std::any&, real_t);
            };
            // Share one dispatch table across all instances holding the same type
            template <typename T>
                inline static constexpr VTable _vtable_for{
                    [](const std::any &value, const real_t x) {
                        return std::any_cast<const T&>(value)(x);
                    },
                    [](const std::any &value, const real_t x) {
                        return std::any_cast<const T&>(value).derivative(x);
                    }
                };
            // Data members
            std::any _value;
            const VTable *_vtable;
    };
    /*
    class Activation {
        private:
            struct Interface {
                virtual ~Interface() = default;
                virtual real_t value(const real_t x) const = 0;
                virtual real_t derivative(const real_t x) const = 0;
                virtual std::unique_ptr<Interface> clone() const = 0;
            };
            template <typename T>
                struct Model final : Interface {
                    T activation;
                    // Store the concrete activation
                    explicit Model(T value) : activation(std::move(value)) {}
                    // Compute the activation value
                    real_t value(const real_t x) const override {
                        return activation(x);
                    }
                    // Compute the activation derivative
                    real_t derivative(const real_t x) const override {
                        return activation.derivative(x);
                    }
                    // Copy the concrete activation
                    std::unique_ptr<Interface> clone() const override {
                        return std::make_unique<Model<T>>(activation);
                    }
                };
        public:
            // Store a copyable activation without depending on its concrete type
            template <typename T>
                requires (not std::same_as<std::remove_cvref_t<T>, Activation>) and ActivationType<T> and std::copy_constructible<T>
                explicit Activation(T activation) : _model(std::make_unique<Model<T>>(std::move(activation))) {}
            // Copy the stored activation
            Activation(const Activation &other) : _model(other._model ? other._model->clone() : nullptr) {}
            // Replace the stored activation with a copy
            Activation& operator=(const Activation &other) {
                _model = other._model ? other._model->clone() : nullptr;
                return *this;
            }
            // Transfer ownership of the stored activation
            Activation(Activation&&) noexcept = default;
            Activation& operator=(Activation&&) noexcept = default;
            // Compute the activation value
            real_t operator()(const real_t x) const {
                if (not _model) throw std::logic_error("Activation::operator(): empty activation");
                return _model->value(x);
            }
            // Compute the activation derivative
            real_t derivative(const real_t x) const {
                if (not _model) throw std::logic_error("Activation::derivative: empty activation");
                return _model->derivative(x);
            }
            // Retrieve the stored activation when its type matches
            template <typename T>
                const T* get_if() const {
                    if constexpr (ActivationType<T> and std::copy_constructible<T>) {
                        const auto model = dynamic_cast<const Model<std::remove_cv_t<T>>*>(_model.get());
                        return model ? &model->activation : nullptr;
                    }
                    else {
                        return nullptr;
                    }
                }
        private:
            // Data members
            std::unique_ptr<Interface> _model;
    };
    */
}
