// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// A recovery point for helper threads that read guest memory or run code that may assert on
// data they guessed (draw preparation, vk_draw_prep.h): a fault or a failed assertion on such a
// thread jumps back to the point instead of being handled or ending the process. Only for code
// that holds no lock and owns no state the jump would leave half changed.

#pragma once

#ifdef __linux__
#include <csetjmp>
#include <csignal>
#include <pthread.h>
#endif

namespace Common::Recover {

#ifdef __linux__
inline thread_local sigjmp_buf* point = nullptr;

/// True on a thread inside a recovery point.
[[nodiscard]] inline bool Active() noexcept {
    return point != nullptr;
}

/// Jumps to the active recovery point (from a signal handler too).
[[noreturn]] inline void Jump() noexcept {
    sigjmp_buf* target = point;
    point = nullptr;
    siglongjmp(*target, 1);
}

/// After a jump out of a fault handler the fault signals are still blocked on this thread (the
/// handler runs with them masked): the next fault would end the process.
inline void UnblockFaultSignals() noexcept {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGSEGV);
    sigaddset(&set, SIGBUS);
    pthread_sigmask(SIG_UNBLOCK, &set, nullptr);
}
#else
[[nodiscard]] inline bool Active() noexcept {
    return false;
}
[[noreturn]] inline void Jump() noexcept {
    __builtin_unreachable();
}
inline void UnblockFaultSignals() noexcept {}
#endif

} // namespace Common::Recover
