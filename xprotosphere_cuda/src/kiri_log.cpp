/***
 * @Author: Jayden Zhang
 * @Date: 2020-09-27 03:01:47
 * @LastEditTime: 2023-02-25 02:21:25
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @Description:
 * @FilePath: \XProtoSphere\xprotosphere_cuda\src\kiri_log.cpp
 */

#include <kiri_log.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace KIRI {
spdlog::logger KiriLog::mLogger("KIRI_LOG");

void KiriLog::Init(std::string path) {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::trace);
  console_sink->set_pattern("%^[%T] %n: %v%$");

  auto file_sink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
  file_sink->set_level(spdlog::level::info);

  mLogger = spdlog::logger("KIRI_LOG", {console_sink, file_sink});
  mLogger.set_level(spdlog::level::trace);
}
} // namespace KIRI
