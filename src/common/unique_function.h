// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace Common {

/// General purpose function wrapper similar to std::function.
/// Unlike std::function, the captured values don't have to be copyable.
/// This class can be moved but not copied.
template <typename ResultType, typename... Args>
class UniqueFunction {
    class CallableBase {
    public:
        virtual ~CallableBase() = default;
        virtual ResultType operator()(Args&&...) = 0;
    };

    template <typename Functor>
    class Callable final : public CallableBase {
    public:
        Callable(Functor&& functor_) : functor{std::move(functor_)} {}
        ~Callable() override = default;

        ResultType operator()(Args&&... args) override {
            return functor(std::forward<Args>(args)...);
        }

    private:
        Functor functor;
    };

public:
    UniqueFunction() = default;

    template <typename Functor>
    UniqueFunction(Functor&& functor)
        : callable{std::make_unique<Callable<Functor>>(std::move(functor))} {
        // Reject lvalues because Callable<F&> would retain a reference that can outlive its target
        static_assert(!std::is_lvalue_reference_v<Functor>,
                      "pass the callable as a temporary or std::move it");
    }

    UniqueFunction& operator=(UniqueFunction&& rhs) noexcept = default;
    UniqueFunction(UniqueFunction&& rhs) noexcept = default;

    UniqueFunction& operator=(const UniqueFunction&) = delete;
    UniqueFunction(const UniqueFunction&) = delete;

    ResultType operator()(Args&&... args) const {
        return (*callable)(std::forward<Args>(args)...);
    }

    explicit operator bool() const noexcept {
        return static_cast<bool>(callable);
    }

private:
    std::unique_ptr<CallableBase> callable;
};

} // namespace Common
