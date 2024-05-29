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
#include <plog/WinApi.h>

namespace plog
{
    template <class Formatter>
    class EventLogAppender : public IAppender
    {
    public:
        EventLogAppender(const wchar_t* sourceName) : m_eventSource(RegisterEventSourceW(NULL, sourceName))
        {
        }

        ~EventLogAppender()
        {
            DeregisterEventSource(m_eventSource);
        }

        virtual void write(const Record& record)
        {
            std::wstring str = Formatter::format(record);
            const wchar_t* logMessagePtr[] = { str.c_str() };

            ReportEventW(m_eventSource, logSeverityToType(record.getSeverity()), static_cast<WORD>(record.getSeverity()), 0, NULL, 1, 0, logMessagePtr, NULL);
        }

    private:
        static WORD logSeverityToType(plog::Severity severity)
        {
            switch (severity)
            {
            case plog::fatal:
            case plog::error:
                return eventLog::kErrorType;

            case plog::warning:
                return eventLog::kWarningType;

            case plog::info:
            case plog::debug:
            case plog::verbose:
            default:
                return eventLog::kInformationType;
            }
        }

    private:
        HANDLE m_eventSource;
    };

    class EventLogAppenderRegistry
    {
    public:
        static bool add(const wchar_t* sourceName, const wchar_t* logName = L"Application")
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            HKEY sourceKey;
            if (0 != RegCreateKeyExW(hkey::kLocalMachine, sourceKeyName.c_str(), 0, NULL, 0, regSam::kSetValue, NULL, &sourceKey, NULL))
            {
                return false;
            }

            const DWORD kTypesSupported = eventLog::kErrorType | eventLog::kWarningType | eventLog::kInformationType;
            RegSetValueExW(sourceKey, L"TypesSupported", 0, regType::kDword, reinterpret_cast<const BYTE*>(&kTypesSupported), sizeof(kTypesSupported));

            const wchar_t kEventMessageFile[] = L"%windir%\\Microsoft.NET\\Framework\\v4.0.30319\\EventLogMessages.dll;%windir%\\Microsoft.NET\\Framework\\v2.0.50727\\EventLogMessages.dll";
            RegSetValueExW(sourceKey, L"EventMessageFile", 0, regType::kExpandSz, reinterpret_cast<const BYTE*>(kEventMessageFile), sizeof(kEventMessageFile) - sizeof(*kEventMessageFile));

            RegCloseKey(sourceKey);
            return true;
        }

        static bool exists(const wchar_t* sourceName, const wchar_t* logName = L"Application")
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            HKEY sourceKey;
            if (0 != RegOpenKeyExW(hkey::kLocalMachine, sourceKeyName.c_str(), 0, regSam::kQueryValue, &sourceKey))
            {
                return false;
            }

            RegCloseKey(sourceKey);
            return true;
        }

        static void remove(const wchar_t* sourceName, const wchar_t* logName = L"Application")
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            RegDeleteKeyW(hkey::kLocalMachine, sourceKeyName.c_str());
            RegDeleteKeyW(hkey::kLocalMachine, logKeyName.c_str());
        }

    private:
        static void getKeyNames(const wchar_t* sourceName, const wchar_t* logName, std::wstring& sourceKeyName, std::wstring& logKeyName)
        {
            const std::wstring kPrefix = L"SYSTEM\\CurrentControlSet\\Services\\EventLog\\";
            logKeyName = kPrefix + logName;
            sourceKeyName = logKeyName + L"\\" + sourceName;
        }
    };
}
