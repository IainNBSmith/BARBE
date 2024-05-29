/*******************************************************************************
 * Copyright (C) 2020 Mohammad Motallebi
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#pragma once
#include <plog/Appenders/IAppender.h>
#include <android/log.h>

namespace plog
{
    template<class Formatter>
    class AndroidAppender : public IAppender
    {
    public:
        AndroidAppender(const char* tag) : m_tag(tag)
        {
        }

        virtual void write(const Record& record)
        {
            std::string str = Formatter::format(record);

            __android_log_print(toPriority(record.getSeverity()), m_tag, "%s", str.c_str());
        }

    private:
        static android_LogPriority toPriority(Severity severity)
        {
            switch (severity)
            {
            case fatal:
                return ANDROID_LOG_FATAL;
            case error:
                return ANDROID_LOG_ERROR;
            case warning:
                return ANDROID_LOG_WARN;
            case info:
                return ANDROID_LOG_INFO;
            case debug:
                return ANDROID_LOG_DEBUG;
            case verbose:
                return ANDROID_LOG_VERBOSE;
            default:
                return ANDROID_LOG_UNKNOWN;
            }
        }

    private:
        const char* const m_tag;
    };
}
