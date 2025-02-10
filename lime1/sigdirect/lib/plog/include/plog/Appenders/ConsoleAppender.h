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
#include <plog/Util.h>
#include <plog/WinApi.h>
#include <iostream>

namespace plog
{
    enum OutputStream
    {
        streamStdOut,
        streamStdErr
    };

    template<class Formatter>
    class ConsoleAppender : public IAppender
    {
    public:
#ifdef _WIN32
#   ifdef _MSC_VER
#       pragma warning(suppress: 26812) //  Prefer 'enum class' over 'enum'
#   endif
        ConsoleAppender(OutputStream outStream = streamStdOut)
            : m_isatty(!!_isatty(_fileno(outStream == streamStdOut ? stdout : stderr)))
            , m_outputStream(outStream == streamStdOut ? std::cout : std::cerr)
            , m_outputHandle()
        {
            if (m_isatty)
            {
                m_outputHandle = GetStdHandle(outStream == streamStdOut ? stdHandle::kOutput : stdHandle::kErrorOutput);
            }
        }
#else
        ConsoleAppender(OutputStream outStream = streamStdOut) 
            : m_isatty(!!isatty(fileno(outStream == streamStdOut ? stdout : stderr))) 
            , m_outputStream(outStream == streamStdOut ? std::cout : std::cerr)
        {}
#endif

        virtual void write(const Record& record)
        {
            util::nstring str = Formatter::format(record);
            util::MutexLock lock(m_mutex);

            writestr(str);
        }

    protected:
        void writestr(const util::nstring& str)
        {
#ifdef _WIN32
            if (m_isatty)
            {
                WriteConsoleW(m_outputHandle, str.c_str(), static_cast<DWORD>(str.size()), NULL, NULL);
            }
            else
            {
                m_outputStream << util::toNarrow(str, codePage::kActive) << std::flush;
            }
#else
            m_outputStream << str << std::flush;
#endif
        }

    private:
#ifdef __BORLANDC__
        static int _isatty(int fd) { return ::isatty(fd); }
#endif

    protected:
        util::Mutex m_mutex;
        const bool  m_isatty;
        std::ostream& m_outputStream;
#ifdef _WIN32
        HANDLE      m_outputHandle;
#endif
    };
}
