/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-25 14:19:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-25 14:24:51
 * @FilePath: \XProtoSphere\xprotosphere_cuda\include\kiri_log.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _KIRI_LOG_H_
#define _KIRI_LOG_H_

#pragma once

// clang-format off
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
// clang-format on

namespace KIRI {
class KiriLog {
public:
  static void Init(std::string path);

  inline static spdlog::logger &GetLogger() { return mLogger; };

private:
  static spdlog::logger mLogger;
};
} // namespace KIRI

#define KIRI_LOG_TRACE(...) ::KIRI::KiriLog::GetLogger().trace(__VA_ARGS__)
#define KIRI_LOG_INFO(...) ::KIRI::KiriLog::GetLogger().info(__VA_ARGS__)
#define KIRI_LOG_DEBUG(...) ::KIRI::KiriLog::GetLogger().debug(__VA_ARGS__)
#define KIRI_LOG_WARN(...) ::KIRI::KiriLog::GetLogger().warn(__VA_ARGS__)
#define KIRI_LOG_ERROR(...) ::KIRI::KiriLog::GetLogger().error(__VA_ARGS__)
#define KIRI_LOG_FATAL(...) ::KIRI::KiriLog::GetLogger().fatal(__VA_ARGS__)

#endif